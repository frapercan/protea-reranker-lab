"""DeepGO-SE+k-WTA+554k as a pure RESCORER of the DEPLOYED candidate pool.

The separability result (measure_avg.json): on the reachable pool tail, SE out-separates the deployed
reranker per protein -- LK-BPO SE 0.867 vs reranker 0.829; PK-BPO SE 0.626 vs reranker 0.491 (chance).
The prior CONVERT test only used SE as a GENERATOR (adding out-of-pool candidates -> floods, all NO-GO).
It never tested SE as a pure RESCORER of the IN-POOL candidates. That is this test.

ONE variable = the score. The candidate set is FROZEN to the deployed submission (percut pool). Arms:
  A         deployed reranker scores                       -> reproduce anchor LK 0.31323 / PK 0.14351
  B_SE      each in-pool BP (prot,term) rescored by SE avg  (SE=0 for the <0.2% BP terms outside the
            19,723-term SE vocab; non-BP rows and non-target proteins left verbatim, irrelevant to BP)
  B_blend   rank-average of reranker rank and SE rank over BP rows, renormalised to [0,1]
  CONTROL   B_SE's SE scores shuffled across the BP rows (same multiset) -> must be strongly negative

Metric: BP f_micro_w in the TRUE board frame (lab obo+IA, prop=fill norm=cafa no_orphans toi, PK -known),
cafa_eval under PROTEA/.venv, verbatim driver from deepgose_measure.py. Delta vs the reproduced anchor,
with a paired protein-resample bootstrap CI (same resample for A and B: the per-protein variance cancels).

LEAKAGE: SE model trained on t0<=v225; blind window v227->v230 -> temporally clean by construction. The
blend is a FIXED 50/50 rank-average, no weight fit on anything (and no SE scores exist for past proteins
to fit on), so no eval-window tuning is possible. Verified, not asserted.
"""
import json, collections, time, subprocess, tempfile, sys
from pathlib import Path
import numpy as np, pandas as pd

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
PREDDIR = R / "percut_rerank/predictions"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
SE = ROOT / "storage/deepgose_kwta"
OUT = ROOT / "storage/deepgose_rescore"
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
ANCHOR = {"LK-BPO": 0.31323, "PK-BPO": 0.14351}
NBOOT = 100
SEED = 42

# ---- obo: namespace + alt only (we do not need ancestors here; scoring is per in-pool pair) ----
ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}

# ---- cafaeval driver (true frame, verbatim from deepgose_measure.py) ----
DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs = cafa_eval("{obo}", "{pd}", "{gt}", ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, toi_file="{toi}", exclude={known}, max_terms=None, th_step=0.01,
    n_cpu=6, weighted_only=False)
out = {{}}
for k, v in dfs.items(): out[k] = v.reset_index().to_dict(orient="records")
json.dump(out, open("{o}", "w"), default=str)
'''

def _run_cafa(rows, gt_path, known):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows: fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        knownrepr = f'"{known}"' if known else "None"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=gt_path, ia=IA_F, toi=TOI,
                                     known=knownrepr, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None

def rank01(x):
    """ordinal rank of x mapped to [0,1] (ties broken by position; adequate for a blend)."""
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return order / (len(x) - 1) if len(x) > 1 else np.full(len(x), 0.5)

def build_arms(cell, cat):
    """Return (base, B_SE, B_blend, B_ctrl) row-lists over the SAME frozen candidate set, plus meta."""
    Z = np.load(SE / f"se_scores_{cell.split('-')[0]}.npz", allow_pickle=True)
    prots = Z["proteins"].tolist(); terms = [alt.get(g, g) for g in Z["terms"].tolist()]
    S = Z["avg"]
    tpos = {g: j for j, g in enumerate(terms)}; pidx = {p: i for i, p in enumerate(prots)}
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    P = dep.prot.to_numpy(); G = dep.term.to_numpy(); RR = dep.score.to_numpy(dtype=np.float64)
    isbp = np.array([g in BP for g in G])
    have_se = np.array([isbp[i] and P[i] in pidx and G[i] in tpos for i in range(len(P))])
    se = np.zeros(len(P), np.float64)
    idx = np.where(have_se)[0]
    se[idx] = [S[pidx[P[i]], tpos[G[i]]] for i in idx]
    meta = {"rows": int(len(P)), "bp_rows": int(isbp.sum()),
            "bp_rows_with_se": int(have_se.sum()),
            "bp_rows_missing_se": int((isbp & ~have_se).sum()),
            "n_targets_in_se": len(prots)}
    # A: verbatim deployed
    base = list(zip(P.tolist(), G.tolist(), RR.tolist()))
    # B_SE: BP rows -> SE (0 outside SE vocab); non-BP rows verbatim
    se_score = np.where(isbp, se, RR)
    B_SE = list(zip(P.tolist(), G.tolist(), se_score.tolist()))
    # B_blend: rank-average over BP rows only (non-BP verbatim)
    bp_mask = isbp
    rr_r = np.zeros(len(P)); se_r = np.zeros(len(P))
    rr_r[bp_mask] = rank01(RR[bp_mask]); se_r[bp_mask] = rank01(se[bp_mask])
    blend = np.where(bp_mask, 0.5 * (rr_r + se_r), RR)
    B_blend = list(zip(P.tolist(), G.tolist(), blend.tolist()))
    # CONTROL: SE scores on BP rows shuffled across BP rows (same multiset)
    rng = np.random.default_rng(SEED)
    ctrl = RR.copy()
    bp_idx = np.where(bp_mask)[0]
    shuf = se[bp_idx].copy(); rng.shuffle(shuf)
    ctrl[bp_idx] = shuf
    B_ctrl = list(zip(P.tolist(), G.tolist(), ctrl.tolist()))
    return base, B_SE, B_blend, B_ctrl, meta, P

results = {"design": "pure RESCORE of the frozen deployed pool; one variable = the score",
           "frame": "TRUE board (lab obo+IA, prop=fill norm=cafa no_orphans toi, PK -known)",
           "nboot": NBOOT}
CELLS = [("LK-BPO", "lk", str(GTDIR / "groundtruth_LK.tsv"), None),
         ("PK-BPO", "pk", str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))]
ARMS = {}  # cell -> (base, B_SE, B_blend, P, gt_fn, known)

# ================= PHASE 1: point estimates on the FULL frozen set (fast, decisive) =================
for cell, cat, gt_fn, known in CELLS:
    log(f"===== {cell} (point estimates) =====")
    base, B_SE, B_blend, B_ctrl, meta, P = build_arms(cell, cat)
    log(f"  {meta}")
    fa = _run_cafa(base, gt_fn, known)       # pass original gt file verbatim
    log(f"  A (deployed reranker): {fa}  (anchor {ANCHOR[cell]})")
    repro = fa is not None and abs(fa - ANCHOR[cell]) < 0.001
    cell_res = {"anchor": ANCHOR[cell], "A_reproduced": fa, "PRECONDITION_anchor_ok": bool(repro),
                "meta": meta}
    if not repro:
        log("  PRECONDITION FAILED: A does not reproduce the anchor. VOID this cell.")
        results[cell] = cell_res
        json.dump(results, open(OUT / "rescore.json", "w"), indent=1); continue
    f_se = _run_cafa(B_SE, gt_fn, known)
    f_bl = _run_cafa(B_blend, gt_fn, known)
    f_ct = _run_cafa(B_ctrl, gt_fn, known)
    d_se = round(f_se - fa, 5) if f_se is not None else None
    d_bl = round(f_bl - fa, 5) if f_bl is not None else None
    d_ct = round(f_ct - fa, 5) if f_ct is not None else None
    log(f"  B_SE {f_se} (d {d_se:+}) | B_blend {f_bl} (d {d_bl:+}) | CONTROL_random {f_ct} (d {d_ct:+})")
    cell_res.update({"B_SE": f_se, "delta_B_SE": d_se, "B_blend": f_bl, "delta_B_blend": d_bl,
                     "CONTROL_random": f_ct, "delta_CONTROL_random": d_ct})
    results[cell] = cell_res
    ARMS[cell] = (base, B_SE, B_blend, P, gt_fn, known)
    json.dump(results, open(OUT / "rescore.json", "w"), indent=1)

# ================= PHASE 2: paired protein-resample bootstrap CI for B_SE and B_blend =================
for cell, cat, gt_fn, known in CELLS:
    if cell not in ARMS: continue
    base, B_SE, B_blend, P, gt_fn, known = ARMS[cell]
    cell_res = results[cell]
    log(f"===== {cell} (bootstrap) =====")
    # read the original gt file lines so a resample writes a format-identical restricted gt
    gt_lines = Path(gt_fn).read_text().splitlines()
    header = gt_lines[0]
    prot_of = [ln.split("\t", 1)[0] for ln in gt_lines[1:]]
    body = gt_lines[1:]
    gt_prots = sorted(set(prot_of) & set(P.tolist()))
    baseD = {(p, g): v for p, g, v in base}
    seD = {(p, g): v for p, g, v in B_SE}
    blD = {(p, g): v for p, g, v in B_blend}
    keys = list(baseD.keys())
    def cafa_sub(D, samp, tmpgt):
        rows = [(p, g, D[(p, g)]) for (p, g) in keys if p in samp]
        return _run_cafa(rows, tmpgt, known)
    rng = np.random.default_rng(SEED)
    boot_se, boot_bl = [], []
    with tempfile.TemporaryDirectory() as td:
        for b in range(NBOOT):
            samp = set(rng.choice(gt_prots, size=len(gt_prots), replace=True).tolist())
            tmpgt = str(Path(td) / f"gt_{b}.tsv")
            with open(tmpgt, "w") as fh:
                fh.write(header + "\n")
                for ln, pr in zip(body, prot_of):
                    if pr in samp: fh.write(ln + "\n")
            xa = cafa_sub(baseD, samp, tmpgt)
            xse = cafa_sub(seD, samp, tmpgt)
            xbl = cafa_sub(blD, samp, tmpgt)
            if xa is not None and xse is not None: boot_se.append(round(xse - xa, 5))
            if xa is not None and xbl is not None: boot_bl.append(round(xbl - xa, 5))
            if (b + 1) % 10 == 0:
                log(f"  boot {b+1}/{NBOOT}: B_SE-A mean {np.mean(boot_se):+.5f} | "
                    f"B_blend-A mean {np.mean(boot_bl):+.5f}")
                cell_res["bootstrap_progress"] = {"done": b + 1,
                    "B_SE": {"mean": round(float(np.mean(boot_se)), 5)},
                    "B_blend": {"mean": round(float(np.mean(boot_bl)), 5)}}
                json.dump(results, open(OUT / "rescore.json", "w"), indent=1)
    def ci(bs):
        return {"n": len(bs), "mean": round(float(np.mean(bs)), 5),
                "ci95": [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                "excludes_zero_positive": bool(np.percentile(bs, 2.5) > 0)}
    cell_res.pop("bootstrap_progress", None)
    cell_res["bootstrap_B_SE"] = ci(boot_se)
    cell_res["bootstrap_B_blend"] = ci(boot_bl)
    cell_res["bootstrap_note"] = ("paired protein resample (with replacement, deduped to the restricted "
        "gt/pred subset so PK -known matching is preserved); same resample scores A and B (paired)")
    log(f"  CI B_SE {cell_res['bootstrap_B_SE']['ci95']} | B_blend {cell_res['bootstrap_B_blend']['ci95']}")
    results[cell] = cell_res
    json.dump(results, open(OUT / "rescore.json", "w"), indent=1)

# ---- verdict ----
def conv(cell):
    c = results.get(cell, {})
    return bool(c.get("delta_B_SE", -1) is not None and c.get("delta_B_SE", -1) > 0) or \
           bool(c.get("delta_B_blend", -1) is not None and c.get("delta_B_blend", -1) > 0)
results["verdict"] = {
    "LK_converts": conv("LK-BPO"), "PK_converts": conv("PK-BPO"),
    "GO": bool(conv("LK-BPO") or conv("PK-BPO")),
    "one_line": None}
json.dump(results, open(OUT / "rescore.json", "w"), indent=1)
log("wrote rescore.json")
print(json.dumps(results, indent=1))
