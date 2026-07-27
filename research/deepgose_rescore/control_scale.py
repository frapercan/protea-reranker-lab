"""The fixed-score control the campaign requires (SCALE is a free lever >=0.088; no cafaeval delta is
trustworthy without it). B_SE changes BOTH the ordering AND the score distribution vs the reranker.
This isolates the two:

  A            reranker raw scores (the anchor)
  A_rankonly   reranker score -> rank01 (SAME ordering as A, uniform [0,1] distribution)
  B_SE         SE avg scores (SE ordering + SE distribution)          -- from rescore.py
  B_SE_rank    SE -> rank01 (SE ordering, uniform distribution)

If B_SE > A_rankonly, SE's ORDERING beats the reranker's ordering (real signal, not a grid/scale
artefact). If B_SE ~= A_rankonly, the gain was pure rescaling of the reranker's own order.
"""
import json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
PREDDIR = R / "percut_rerank/predictions"; GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
SE = ROOT / "storage/deepgose_kwta"; OUT = ROOT / "storage/deepgose_rescore"
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
ANCHOR = {"LK-BPO": 0.31323, "PK-BPO": 0.14351}

ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs = cafa_eval("{obo}", "{pd}", "{gt}", ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, toi_file="{toi}", exclude={known}, max_terms=None, th_step=0.01,
    n_cpu=4, weighted_only=False)
out = {{}}
for k, v in dfs.items(): out[k] = v.reset_index().to_dict(orient="records")
json.dump(out, open("{o}", "w"), default=str)
'''
def cafa(rows, gt_path, known):
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
            print(r.stderr[-1000:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None

def rank01(x):
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return order / (len(x) - 1) if len(x) > 1 else np.full(len(x), 0.5)

res = {}
for cell, cat, gt_fn, known in [
    ("LK-BPO", "lk", str(GTDIR / "groundtruth_LK.tsv"), None),
    ("PK-BPO", "pk", str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))]:
    log(f"===== {cell} =====")
    Z = np.load(SE / f"se_scores_{cell.split('-')[0]}.npz", allow_pickle=True)
    prots = Z["proteins"].tolist(); terms = [alt.get(g, g) for g in Z["terms"].tolist()]
    S = Z["avg"]; tpos = {g: j for j, g in enumerate(terms)}; pidx = {p: i for i, p in enumerate(prots)}
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    P = dep.prot.to_numpy(); G = dep.term.to_numpy(); RR = dep.score.to_numpy(dtype=np.float64)
    isbp = np.array([g in BP for g in G])
    se = np.zeros(len(P))
    for i in np.where(isbp)[0]:
        if P[i] in pidx and G[i] in tpos: se[i] = S[pidx[P[i]], tpos[G[i]]]
    # A_rankonly: reranker order, uniform dist (BP rows only; non-BP verbatim)
    rr_rank = RR.copy(); rr_rank[isbp] = rank01(RR[isbp])
    se_rank = RR.copy(); se_rank[isbp] = rank01(se[isbp])
    fa = cafa(list(zip(P, G, RR)), gt_fn, known)
    f_ar = cafa(list(zip(P, G, rr_rank)), gt_fn, known)
    f_sr = cafa(list(zip(P, G, se_rank)), gt_fn, known)
    res[cell] = {"anchor": ANCHOR[cell], "A": fa,
                 "A_rankonly": f_ar, "delta_A_rankonly": round(f_ar - fa, 5) if f_ar else None,
                 "B_SE_rank": f_sr, "delta_B_SE_rank": round(f_sr - fa, 5) if f_sr else None}
    log(f"  A {fa} | A_rankonly {f_ar} (d {res[cell]['delta_A_rankonly']:+}) | "
        f"B_SE_rank {f_sr} (d {res[cell]['delta_B_SE_rank']:+})")
    json.dump(res, open(OUT / "control_scale.json", "w"), indent=1)
log("done")
print(json.dumps(res, indent=1))
