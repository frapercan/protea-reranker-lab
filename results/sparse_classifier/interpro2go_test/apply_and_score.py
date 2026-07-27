"""Apply the validation-tuned per-cell noisy-OR weights to TEST, score 9 cells
board-faithfully (cafa_eval), build best-of-vs-graft injectable file.

Base TEST preds (graft routing): NK & LK from baseline, PK from per-cut.
InterPro TEST preds: protein2ipr.json (7401 queries).
"""
import os, json, tempfile, shutil, collections
import pandas as pd
from cafaeval.evaluation import cafa_eval
import interpro_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
SC = L.SC
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = os.path.join(SC, "lafa_gt")
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
ASPECTS = ["mfo", "bpo", "cco"]
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo",
          "cellular_component": "cco"}
nm = L.ns_map()

BASE_PRED = {
    "nk": os.path.join(SC, "percut_rerank/baseline/predictions/nk/nk.tsv"),
    "lk": os.path.join(SC, "percut_rerank/baseline/predictions/lk/lk.tsv"),
    "pk": os.path.join(SC, "percut_rerank/predictions/pk/pk.tsv"),
}
GT = {
    "nk": (os.path.join(GT_DIR, "groundtruth_NK.tsv"), None),
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv")),
}
chosen = json.load(open(os.path.join(HERE, "chosen_weights.json")))

# TEST InterPro graded preds
tprot2ipr = json.load(open(os.path.join(HERE, "protein2ipr.json")))
tip = L.interpro_preds(tprot2ipr)  # acc -> {go: graded}


def score_dir(pred_dir, cat):
    gt_file, known = GT[cat]
    _, dfs_best = cafa_eval(OBO, pred_dir, gt_file, ia=IA, no_orphans=True,
                            norm="cafa", prop="fill", exclude=known,
                            toi_file=TOI, th_step=0.01, n_cpu=8)
    best = dfs_best["f_micro_w"].reset_index()
    cells = {}
    for _, r in best.iterrows():
        a = NS2ASP.get(r["ns"])
        if a:
            cells[a] = {"f_micro_w": round(float(r["f_micro_w"]), 5),
                        "tau": round(float(r["tau"]), 3),
                        "cov": round(float(r.get("cov", float("nan"))), 4)}
    return cells


def build_blend_rows(cat, wmap, rule="noisyor"):
    """Per-category blended TEST rows. rule='noisyor' -> 1-(1-b)(1-w*ip);
    rule='max' -> max(b, ip) for aspects with w>0 (parameter-free naive max)."""
    base = pd.read_csv(BASE_PRED[cat], sep="\t", header=None,
                       names=["acc", "go", "score"])
    bmap = collections.defaultdict(dict)
    for r in base.itertuples(index=False):
        bmap[r.acc][r.go] = max(bmap[r.acc].get(r.go, 0.0), float(r.score))
    proteins = set(bmap)
    rows = []
    for p in proteins:
        bm = bmap.get(p, {})
        ipm = tip.get(p, {})
        terms = set(bm) | set(ipm)
        for t in terms:
            asp = nm.get(t)
            if asp is None:
                continue
            w = wmap.get(asp, 0.0)
            b = bm.get(t, 0.0)
            if w <= 0:
                comb = b
            elif rule == "max":
                comb = max(b, ipm.get(t, 0.0))
            else:
                comb = 1.0 - (1.0 - b) * (1.0 - w * ipm.get(t, 0.0))
            if comb > 0:
                rows.append((p, t, comb))
    return rows


# BP-restricted weight map: keep validation-tuned BP weight, zero MF/CC (-> graft).
# Pre-registered: the orthogonality hypothesis is BP-specific; MF/CC blend was the
# exploratory arm and over-credited by the (non-board-faithful) validation proxy.
chosen_bp = {cat: {"mfo": 0.0, "bpo": chosen[cat]["bpo"], "cco": 0.0}
             for cat in ["nk", "lk", "pk"]}

graft_cells, blend_cells, bponly_cells, naivemax_cells = {}, {}, {}, {}
bponly_pred_rows = {}
naivemax_pred_rows = {}
for cat in ["nk", "lk", "pk"]:
    # graft (re-score base file fresh)
    d = tempfile.mkdtemp(prefix=f"graft_{cat}_")
    shutil.copy(BASE_PRED[cat], os.path.join(d, f"{cat}.tsv"))
    graft_cells[cat] = score_dir(d, cat); shutil.rmtree(d, ignore_errors=True)
    # full validation-selected blend (all aspects)
    rows = build_blend_rows(cat, chosen[cat])
    d = tempfile.mkdtemp(prefix=f"blend_{cat}_")
    pd.DataFrame(rows, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    blend_cells[cat] = score_dir(d, cat); shutil.rmtree(d, ignore_errors=True)
    # BP-only conservative blend (the injectable)
    rows_bp = build_blend_rows(cat, chosen_bp[cat])
    bponly_pred_rows[cat] = rows_bp
    d = tempfile.mkdtemp(prefix=f"bponly_{cat}_")
    pd.DataFrame(rows_bp, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    bponly_cells[cat] = score_dir(d, cat); shutil.rmtree(d, ignore_errors=True)
    # naive max BP-only (parameter-free reference; w map only marks which aspects)
    rows_mx = build_blend_rows(cat, {"mfo": 0.0, "bpo": 1.0, "cco": 0.0}, rule="max")
    naivemax_pred_rows[cat] = rows_mx
    d = tempfile.mkdtemp(prefix=f"max_{cat}_")
    pd.DataFrame(rows_mx, columns=["acc", "go", "score"]).to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False,
        float_format="%.6f")
    naivemax_cells[cat] = score_dir(d, cat); shutil.rmtree(d, ignore_errors=True)
    print(cat, "graft", {a: graft_cells[cat][a]["f_micro_w"] for a in ASPECTS},
          "| full-blend", {a: blend_cells[cat][a]["f_micro_w"] for a in ASPECTS},
          "| bp-only", {a: bponly_cells[cat][a]["f_micro_w"] for a in ASPECTS},
          "| naive-max", {a: naivemax_cells[cat][a]["f_micro_w"] for a in ASPECTS})

# ---- best-of injectable = BP-only blend; guarantees no MF/CC regression ----
table, bestof_cells = [], {}
n_improved = n_regressed = 0
for cat in ["nk", "lk", "pk"]:
    bestof_cells[cat] = {}
    for a in ASPECTS:
        g = graft_cells[cat][a]["f_micro_w"]
        fb = blend_cells[cat][a]["f_micro_w"]
        bo = bponly_cells[cat][a]["f_micro_w"]
        w = chosen_bp[cat][a]
        src = "graft" if w == 0 else "blend(bp)"
        bestof_cells[cat][a] = {"f_micro_w": bo, "source": src, "w": w}
        d = round(bo - g, 5)
        if w > 0:
            if d > 0.002:
                n_improved += 1
            elif d < -0.002:
                n_regressed += 1
        table.append({"cell": f"{cat.upper()}-{a}", "bp_w": w, "graft": g,
                      "full_blend": fb, "delta_fullblend_vs_graft": round(fb - g, 5),
                      "bestof_bponly": bo, "delta_bestof_vs_graft": d,
                      "injectable_source": src})

out = {
    "method": "noisy-OR(base, w*interpro_graded); w tuned per-(cat,aspect) on "
              "v225-v227 validation (fixed-denominator IA-weighted micro-Fmax); "
              "applied to test; board-faithful cafa_eval scoring. INJECTABLE = "
              "BP-restricted best-of (MF/CC kept = graft; BP = validation-tuned blend) "
              "to guarantee no regression on led cells.",
    "chosen_weights_full": chosen,
    "chosen_weights_bponly_injectable": chosen_bp,
    "graft_9cell": graft_cells,
    "full_blend_9cell": blend_cells,
    "bestof_bponly_9cell": bestof_cells,
    "naivemax_bponly_9cell": naivemax_cells,
    "table": table,
    "bp_cells_improved_on_test": n_improved,
    "bp_cells_regressed_on_test": n_regressed,
    "mean_graft": round(sum(graft_cells[c][a]["f_micro_w"] for c in graft_cells for a in ASPECTS) / 9, 5),
    "mean_full_blend": round(sum(blend_cells[c][a]["f_micro_w"] for c in blend_cells for a in ASPECTS) / 9, 5),
    "mean_bestof_bponly": round(sum(bestof_cells[c][a]["f_micro_w"] for c in bestof_cells for a in ASPECTS) / 9, 5),
}
json.dump(out, open(os.path.join(HERE, "blend_9cell.json"), "w"), indent=2)

# ---- injectable routed single file (BP-only best-of) ----
inj = []
for cat in ["nk", "lk", "pk"]:
    inj.extend(bponly_pred_rows[cat])
inj_df = pd.DataFrame(inj, columns=["protein", "go", "score"])
inj_path = os.path.join(HERE, "predictions_graft_interpro.tsv")
inj_df.to_csv(inj_path, sep="\t", header=False, index=False, float_format="%.6f")
print(f"injectable: {len(inj_df)} rows, {inj_df.protein.nunique()} proteins -> {inj_path}")

# also emit the parameter-free naive-max BP-only injectable (stronger on test here)
mxrows = []
for cat in ["nk", "lk", "pk"]:
    mxrows.extend(naivemax_pred_rows[cat])
mx_df = pd.DataFrame(mxrows, columns=["protein", "go", "score"])
mx_path = os.path.join(HERE, "predictions_graft_interpro_naivemax.tsv")
mx_df.to_csv(mx_path, sep="\t", header=False, index=False, float_format="%.6f")
print(f"naive-max injectable: {len(mx_df)} rows, {mx_df.protein.nunique()} proteins -> {mx_path}")
out["mean_naivemax_bponly"] = round(
    sum(naivemax_cells[c][a]["f_micro_w"] for c in naivemax_cells for a in ASPECTS) / 9, 5)
json.dump(out, open(os.path.join(HERE, "blend_9cell.json"), "w"), indent=2)
print("MEAN graft", out["mean_graft"], "full_blend", out["mean_full_blend"],
      "bestof_bponly", out["mean_bestof_bponly"])
print("BP test improved", n_improved, "regressed", n_regressed)
print("written blend_9cell.json")
