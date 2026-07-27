"""PHASE 3 adversarial controls: isolate ORDERING from SCALE/VOLUME on the best positive arm.

The constant-score control conflates ordering with volume. Two sharper controls:
  PERM      : take the winning blend's exact BP score multiset, permute it across BP candidates.
              Same distribution, same scale, same #submitted at every tau; ordering destroyed.
              If f stays high -> the gain is scale/volume, NOT the model's ordering.
  SCALEXFER : give the DEPLOYED ordering the blend's score distribution (rank-match deployed BP
              ranks onto the sorted blend values). Blend's scale + deployed's ordering.
              If f ~ blend -> the gain is the SCALE reshape (the known 0.088 free lever), not the model.
Also a paired protein bootstrap on (blend - incumbent) at the blend's best tau.
"""
import json, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from cafaeval.evaluation import cafa_eval

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
PREDDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/regen_headline"
GT = {"lk": (str(GTDIR / "groundtruth_LK.tsv"), None),
      "pk": (str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))}
ns = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
BP = {t for t, n in ns.items() if n == "biological_process"}

# which arm won each cell (set from phase2 results)
ph2 = json.load(open(OUT / "phase2_cafaeval.json"))
CFG = {"pk": ("gbdt_known_transition", 1.0), "lk": ("gbdt_known_transition", 1.0)}   # (model, add-weight); refined below


def score_full(pred_df, cat):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "pd"; d.mkdir()
        pred_df[["prot", "term", "score"]].to_csv(d / "p.tsv", sep="\t", header=False, index=False,
                                                  float_format="%.6f")
        gt, known = GT[cat]
        _, dfs = cafa_eval(OBO, str(d), gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                           exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
    b = dfs["f_micro_w"].reset_index(); r = b[b.ns == "biological_process"]
    return (float(r.iloc[0]["f_micro_w"]), float(r.iloc[0]["tau"])) if len(r) else (None, None)


def minmax(x):
    x = np.nan_to_num(np.asarray(x, float)); lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


out = {}
rng = np.random.default_rng(42)
for cat in ["pk", "lk"]:
    # pick the best positive arm from phase2 for this cell
    arms = ph2[cat]["arms"]; inc = ph2[cat]["incumbent_f"]
    bestarm = max(arms.items(), key=lambda kv: (kv[1]["f_micro_w"] or -1)
                  if not kv[0].startswith(("A_", "FIXEDSCORE")) else -1)
    tag = bestarm[0]
    model = "gbdt_known_transition" if "gbdt_known_transition" in tag else "gbdt_known"
    print(f"\n[{time.time()-t0:.0f}s] {cat.upper()} best arm {tag} f={bestarm[1]['f_micro_w']} (inc {inc})", flush=True)

    ph1 = pd.read_parquet(OUT / f"phase1_scores_{cat}.parquet").rename(
        columns={"protein_accession": "prot", "go_term_id": "term"}).drop_duplicates(["prot", "term"])
    full = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    full["isbp"] = full.term.isin(BP)
    full = full.merge(ph1[["prot", "term", model]], on=["prot", "term"], how="left")
    bp = full.isbp.values; dep = full.score.values.astype(float)
    mvec = np.zeros(len(full)); mk = bp & full[model].notna().values
    mvec[mk] = minmax(full.loc[mk, model].values)

    # reconstruct the winning blend scores on BP rows (parse weight/rerank from tag)
    blend = dep.copy()
    if tag.startswith("B_"):
        w = float(tag.split("_w")[-1]); blend[bp] = np.clip(dep[bp] + w * mvec[bp], 0, 1)
    elif tag.startswith("C_"):
        blend[bp] = np.clip(mvec[bp], 0, 1)
    elif tag.startswith("D_"):
        dr = np.zeros(len(full)); dr[bp] = pd.Series(dep[bp]).rank().values / bp.sum()
        mr = np.zeros(len(full)); mr[bp] = pd.Series(mvec[bp]).rank().values / bp.sum()
        blend[bp] = 0.5 * dr[bp] + 0.5 * mr[bp]

    f_inc, t_inc = score_full(full.assign(score=dep), cat)
    f_bl, t_bl = score_full(full.assign(score=blend), cat)
    print(f"    incumbent {f_inc:.5f} (tau {t_inc}) | blend {f_bl:.5f} (tau {t_bl}) delta {f_bl-f_inc:+.5f}", flush=True)

    # PERM: permute blend BP scores across BP candidates (same multiset)
    perm_fs = []
    for s in range(3):
        sc = blend.copy(); idx = np.where(bp)[0]
        sc[idx] = blend[idx][rng.permutation(len(idx))]
        f, tau = score_full(full.assign(score=sc), cat); perm_fs.append(f)
        print(f"    PERM seed{s} {f:.5f} (tau {tau})", flush=True)
    # SCALEXFER: deployed ordering, blend distribution (rank-match)
    idx = np.where(bp)[0]
    sorted_blend = np.sort(blend[idx])
    dep_rank = pd.Series(dep[idx]).rank(method="first").astype(int).values - 1
    sc = blend.copy(); sc[idx] = sorted_blend[dep_rank]
    f_sx, t_sx = score_full(full.assign(score=sc), cat)
    print(f"    SCALEXFER (deployed order + blend dist) {f_sx:.5f} (tau {t_sx})", flush=True)

    out[cat] = {"best_arm": tag, "incumbent": round(f_inc, 5), "blend": round(f_bl, 5),
                "blend_delta": round(f_bl - f_inc, 5),
                "perm_control_f": [round(x, 5) for x in perm_fs],
                "perm_control_mean_delta_vs_incumbent": round(float(np.mean(perm_fs)) - f_inc, 5),
                "scalexfer_control_f": round(f_sx, 5),
                "scalexfer_delta_vs_incumbent": round(f_sx - f_inc, 5),
                "interpretation_note": "If PERM or SCALEXFER ~ blend, the gain is SCALE/volume not "
                                       "the model's ordering."}
    json.dump(out, open(OUT / "phase3_controls.json", "w"), indent=2)

print(f"\n[{time.time()-t0:.0f}s] DONE"); print(json.dumps(out, indent=2))
