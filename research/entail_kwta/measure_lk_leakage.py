"""The LK-BPO +0.0207 rescore is a SUSPECT. Diagnosis: LK has no -known exclusion, 100% of LK eval
proteins are in the SE training corpus, and 47% of LK ground-truth BP terms were ALREADY v227 training
labels. So SE reorders the pool to surface terms it MEMORIZED, and LK (unlike PK) lets them count.

Proof: build the LK analog of groundtruth_PK_known (each LK protein's v227 t0 BP labels, propagated)
and re-measure the anchor + SE rescore in that LEAKAGE-CLEAN frame. If the +0.0207 flips negative,
the LK conversion was memorization, not a real conversion -- mirroring PK's -0.0352.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
KW = ROOT / "storage/deepgose_kwta"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
OUT = ROOT / "storage/entail_kwta"
LK_POOL = R / "percut_rerank/predictions/lk/lk.tsv"
GT_LK = str(GTDIR / "groundtruth_LK.tsv")

par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r

# LK analog of groundtruth_PK_known: each LK protein's v227 t0 BP labels, propagated
lab = json.load(open(R / "clf_labels_big.json"))
gt = pd.read_csv(GT_LK, sep="\t"); gt = gt[gt.aspect == "P"].copy()
lk_prots = sorted(gt.EntryID.unique())
# BP-only gt so gt and the (BP-only) exclude file share namespaces (cafa_eval scores ns independently;
# the BP f_micro_w is identical to the full-file BP slice -- verified by the raw-frame anchor 0.31323).
GT_LK_BP = str(OUT / "groundtruth_LK_bp.tsv")
gt[["EntryID", "term", "aspect"]].to_csv(GT_LK_BP, sep="\t", index=False)
lk_known = OUT / "groundtruth_LK_known_v227.tsv"
with lk_known.open("w") as fh:
    fh.write("EntryID\tterm\taspect\n")
    for p in lk_prots:
        known = set()
        for g in lab.get(p, ()):
            if g in BP: known |= anc(g)
        for g in known:
            fh.write(f"{p}\t{g}\tP\n")
log(f"built LK v227-known exclusion: {lk_prots.__len__()} proteins")

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
def cafa(rows, gt_file, known):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows: fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        knownrepr = f'"{known}"' if known else "None"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=gt_file, ia=IA_F, toi=TOI,
                                     known=knownrepr, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None

# rescore (rank-match reranker score multiset in SE order over SE-scored BP rows)
Z = np.load(KW / "se_scores_LK.npz", allow_pickle=True)
prots = Z["proteins"].tolist(); terms = [alt.get(g, g) for g in Z["terms"].tolist()]
S = Z["avg"]; tpos = {g: j for j, g in enumerate(terms)}; pidx = {p: i for i, p in enumerate(prots)}
pool = pd.read_csv(LK_POOL, sep="\t", header=None, names=["p", "t", "s"])
pool["tprim"] = pool["t"].map(lambda g: alt.get(g, g))
new = pool["s"].to_numpy(dtype=np.float64).copy()
bp_has = []; se_vals = []
for i, (p, tp) in enumerate(zip(pool["p"], pool["tprim"])):
    if tp in BP:
        ii = pidx.get(p); jj = tpos.get(tp)
        if ii is not None and jj is not None:
            bp_has.append(i); se_vals.append(float(S[ii, jj]))
bp_has = np.array(bp_has); se_vals = np.array(se_vals)
ladder = np.sort(new[bp_has])[::-1]
rank = np.argsort(-se_vals, kind="mergesort")
new_se = new.copy(); new_se[bp_has[rank]] = ladder
base = list(zip(pool["p"], pool["t"], pool["s"].astype(float)))
rescore = list(zip(pool["p"], pool["t"], new_se))

res = {}
for frame, known in [("raw_no_exclusion", None), ("leakage_clean_v227known_excluded", str(lk_known))]:
    fa = cafa(base, GT_LK_BP, known)
    fb = cafa(rescore, GT_LK_BP, known)
    res[frame] = {"anchor": fa, "rescore_SE": fb,
                  "delta": round(fb - fa, 5) if (fa and fb) else None}
    log(f"  {frame}: anchor {fa} | rescore {fb} | delta {res[frame]['delta']:+}"
        if res[frame]['delta'] is not None else f"  {frame}: FAILED")
json.dump(res, open(OUT / "lk_leakage.json", "w"), indent=1)
log("WROTE lk_leakage.json")
print(json.dumps(res, indent=1))
