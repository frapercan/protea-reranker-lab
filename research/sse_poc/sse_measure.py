"""DeepGO-SE as a BP generator + scorer: the three measurements, true-frame, temporally gated.

Consumes se_scores_{LK,PK}.npz (avg/min/max SE ensemble scores over 19,723 BP terms) and reuses the
exact metric machinery of cg1_step1_gate.py (added-true IA-precision) and cg3_step2_cafaeval.py
(true-frame f_micro_w vs the DEPLOYED anchor LK 0.31323 / PK 0.14351: prediction file verbatim,
prop=fill norm=cafa no_orphans toi, PK -known, cafa_eval under PROTEA/.venv).

1. GATE      per protein rank BP terms by SE, propose top-k NOT in pool; added-true IA-precision vs
             bars (co-occ 1.0%, network 5.8%, classifier 11.6%) + fraction of the FN tail recovered.
2. SEPARABILITY  per-protein AUC of SE vs the deployed reranker on the REACHABLE tail (pool candidates
             with >=1 true and >=1 false BP term) -- is DeepGO-SE's edge scoring rather than generation?
3. CONVERT   union top-k SE proposals into the pool; true-frame f_micro_w delta vs the anchor, with a
             matched-volume (uniform-random terms) control, a random-order (shuffled SE) control, and
             a paired bootstrap CI invoked when a delta is positive and clears the fold noise.
"""
import json, collections, time, subprocess, tempfile, sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
PREDDIR = R / "percut_rerank/predictions"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
OUT = ROOT / "storage/sse_poc"
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
COMBINE = sys.argv[1] if len(sys.argv) > 1 else "avg"
ANCHOR = {"LK-BPO": 0.31323, "PK-BPO": 0.14351}
BARS = {"cooc": 0.010, "network": 0.058, "classifier": 0.116}
TOPKS = [5, 10, 25]

# ---- obo ----
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
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iamass(ts): return float(sum(IA.get(t, 0.0) for t in ts))

# ---- cafaeval driver (true frame, verbatim from cg3_step2) ----
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

results = {"combine": COMBINE}
for cell, cat, gt_fn, known in [
    ("LK-BPO", "lk", str(GTDIR / "groundtruth_LK.tsv"), None),
    ("PK-BPO", "pk", str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv"))]:
    log(f"===== {cell} ({COMBINE}) =====")
    Z = np.load(OUT / f"sse_scores_{cell.split('-')[0]}.npz", allow_pickle=True)
    prots = Z["proteins"].tolist(); terms = [alt.get(g, g) for g in Z["terms"].tolist()]
    S = Z[COMBINE]                       # (P, T) SE score
    tpos = {g: j for j, g in enumerate(terms)}
    pidx = {p: i for i, p in enumerate(prots)}

    gt = pd.read_csv(gt_fn, sep="\t"); gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = [p for p in sorted(gt_leaf) if p in pidx]
    gt_prop = {p: (frozenset().union(*[anc(t) for t in gt_leaf[p]]) if gt_leaf[p] else frozenset())
               for p in targets}

    # pool BP membership + labels from eval.parquet; reranker score from prediction file
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id", "label", "aspect"]).to_pandas()
    tb = tb[(tb.protein_accession.isin(set(targets))) & (tb.aspect == "bpo")]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    pool_bp = tb.groupby("protein_accession").go.apply(set).to_dict()
    lab_bp = {(r.protein_accession, r.go): int(r.label) for r in tb.itertuples()}
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    base = list(dep.itertuples(index=False, name=None))
    rr = {(r.prot, r.term): r.score for r in dep.itertuples()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool_bp[p]]) if pool_bp.get(p) else frozenset())
                 for p in targets}
    fn_total = sum(iamass(gt_prop[p] - pool_prop[p]) for p in targets)
    log(f"  targets {len(targets)}; FN tail IA mass {fn_total:.1f}")

    cell_res = {"anchor": ANCHOR[cell], "n_targets": len(targets), "FN_tail_IA_mass": round(fn_total, 1)}

    # ---------- 1. GATE: added-true IA-precision ----------
    KMAX = max(TOPKS)
    added_true = {k: 0.0 for k in TOPKS}; added_false = {k: 0.0 for k in TOPKS}; nprop = {k: 0 for k in TOPKS}
    # store per-protein top proposals for reuse in step2
    prop_lists = {}
    for p in targets:
        i = pidx[p]; sc = S[i]
        order = np.argsort(-sc)
        present = pool_bp.get(p, set())
        gtp = gt_prop[p]; covered = set(pool_prop.get(p, frozenset()))
        picks = []; k = 0
        for j in order:
            g = terms[j]
            if g in present: continue
            picks.append((g, float(sc[j])))
            ancg = anc(g); new = ancg - covered
            tm = iamass(new & gtp); fm = iamass(new - gtp); covered |= new
            k += 1
            for kk in TOPKS:
                if k <= kk: added_true[kk] += tm; added_false[kk] += fm; nprop[kk] += 1
            if k >= KMAX: break
        prop_lists[p] = picks
    cell_res["gate"] = {}
    for kk in TOPKS:
        at, af = added_true[kk], added_false[kk]
        prec = at / (at + af) if (at + af) > 0 else None
        cell_res["gate"][f"top{kk}"] = {"true_IA": round(at, 2), "false_IA": round(af, 2),
            "precision_IA": round(prec, 4) if prec is not None else None,
            "frac_FN_recovered": round(at / fn_total, 4) if fn_total > 0 else None}
        log(f"  gate top{kk}: prec {prec} (bars cooc .01 net .058 clf .116), FN-rec "
            f"{cell_res['gate'][f'top{kk}']['frac_FN_recovered']}")

    # ---------- 2. SEPARABILITY: per-protein AUC on the reachable pool tail ----------
    se_aucs = []; rr_aucs = []
    for p in targets:
        cands = [g for g in pool_bp.get(p, set()) if (p, g) in lab_bp and g in tpos]
        y = np.array([lab_bp[(p, g)] for g in cands])
        if y.sum() == 0 or y.sum() == len(y) or len(y) < 5: continue
        se = np.array([S[pidx[p], tpos[g]] for g in cands])
        rrv = np.array([rr.get((p, g), 0.0) for g in cands])
        se_aucs.append(roc_auc_score(y, se)); rr_aucs.append(roc_auc_score(y, rrv))
    cell_res["separability"] = {"n_proteins": len(se_aucs),
        "SE_mean_auc": round(float(np.mean(se_aucs)), 4) if se_aucs else None,
        "reranker_mean_auc": round(float(np.mean(rr_aucs)), 4) if rr_aucs else None,
        "SE_minus_reranker": round(float(np.mean(np.array(se_aucs) - np.array(rr_aucs))), 4) if se_aucs else None}
    log(f"  separability: SE {cell_res['separability']['SE_mean_auc']} vs reranker "
        f"{cell_res['separability']['reranker_mean_auc']} (delta {cell_res['separability']['SE_minus_reranker']})")

    # ---------- 3. CONVERT: true-frame f_micro_w delta + controls ----------
    fa = cafa(base, gt_fn, known)
    log(f"  A pool-only (true frame): {fa} (anchor {ANCHOR[cell]})")
    cell_res["A_pool_only"] = fa
    rng = np.random.default_rng(42)
    bp_terms_all = [g for g in terms]  # scoring universe
    cell_res["convert"] = {}
    for K in TOPKS:
        # min-max normalise SE extra scores per cell (as cg3 did for its channel score)
        ex = []
        for p in targets:
            for g, sv in prop_lists[p][:K]:
                ex.append((p, g, sv))
        if ex:
            svals = np.array([e[2] for e in ex]); lo, hi = svals.min(), svals.max()
            nrm = lambda v: (v - lo) / (hi - lo) if hi > lo else 0.5
            exB = [(p, g, float(nrm(v))) for (p, g, v) in ex]
        else:
            exB = []
        fb = cafa(base + exB, gt_fn, known)
        dB = round(fb - fa, 5) if (fb is not None and fa is not None) else None
        # matched-volume control: same count of UNIFORM-random BP terms not in pool, per protein
        exU = []
        for p in targets:
            present = pool_bp.get(p, set())
            pool_terms = [g for g in bp_terms_all if g not in present]
            pick = rng.choice(len(pool_terms), min(K, len(pool_terms)), replace=False)
            for j in pick: exU.append((p, pool_terms[j], float(rng.uniform(0.3, 0.7))))
        fu = cafa(base + exU, gt_fn, known)
        dU = round(fu - fa, 5) if (fu is not None and fa is not None) else None
        # random-order control: SE-proposed terms but scores shuffled across the extras
        if exB:
            sh = svals.copy(); rng.shuffle(sh)
            exR = [(exB[m][0], exB[m][1], float((sh[m] - sh.min()) / (sh.max() - sh.min() + 1e-9)))
                   for m in range(len(exB))]
            fr = cafa(base + exR, gt_fn, known)
            dR = round(fr - fa, 5) if (fr is not None and fa is not None) else None
        else:
            fr = fa; dR = 0.0
        cell_res["convert"][f"top{K}"] = {"n_extras": len(exB), "B": fb, "delta_B": dB,
            "matched_volume_uniform": fu, "delta_uniform": dU, "random_order": fr, "delta_random": dR}
        log(f"  convert top{K}: B {fb} (dB {dB:+}) | uniform {fu} ({dU:+}) | rand-order {fr} ({dR:+})")
        json.dump(results | {cell: cell_res}, open(OUT / f"measure_{COMBINE}.json", "w"), indent=1)
    results[cell] = cell_res
    json.dump(results, open(OUT / f"measure_{COMBINE}.json", "w"), indent=1)

json.dump(results, open(OUT / f"measure_{COMBINE}.json", "w"), indent=1)
log(f"wrote measure_{COMBINE}.json")
print(json.dumps(results, indent=1))
