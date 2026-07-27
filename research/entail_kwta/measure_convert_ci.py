"""CLEAN semantic-entailment over k-WTA: the CONVERSION question, true-frame, with CIs.

The clean minimal EL-embedding ensemble over our learned k-WTA codes (storage/deepgose_kwta:
DeepGOModel MLP tower -> point in the GO-ball space, ELEmbedding nf1..nf4 axiom loss over the t0
2025-07-22 GO EL normal forms, 5 independently-seeded models, semantic-entailment score = ensemble
consensus; NO GAT/DGL, NO MF-preds, NO deepgo2 harness) already has, in the TRUE board frame:
  (a) SEPARABILITY  PK-BPO SE AUC 0.6263 vs reranker 0.491 ; LK-BPO 0.8667 vs 0.8294  (measure_avg.json)
  (b) CONVERSION as a GENERATOR (union top-k SE proposals into the pool): all deltas NEGATIVE, with
      matched-volume + random-order controls (measure_avg.json).

This script adds the two pieces the generator run did not carry:
  1. RESCORE mode -- the cleanest isolation of "does the 0.626 separability convert?". SE avg is a
     sparse sigmoid (median 2e-4), on a different scale from the reranker, so a raw rescore would just
     drop everything under any tau. Instead we RANK-MATCH: keep the reranker's exact score multiset
     over the SE-scored BP pool rows and re-ASSIGN it in SE's order (highest SE gets the highest
     reranker score). By construction every tau submits the IDENTICAL number of terms as the anchor;
     the ONLY thing that changes is WHICH terms -- a perfectly volume-matched test of whether SE's
     ordering separates true from false better than the reranker's, in the metric that decides.
       A  = deployed pool, reranker order              (the anchor)
       B  = same rows, reranker score ladder in SE order
       R  = same ladder in RANDOM order                (control)
  2. Paired protein-bootstrap CI (measure_pk machinery, exact IA-weighted micro-F decomposition at a
     fixed tau, PK -known) on the rescore delta AND on the best generator arm, so a negative verdict
     is a CI that excludes zero, not a bare point.

DISCIPLINE: f_micro_w true-frame (prop=fill norm=cafa no_orphans toi, PK exclude=groundtruth_PK_known)
under the temporal gate (labels v227=t0, blind eval v227->v230) vs the DEPLOYED anchor DECIDES;
AUC is diagnostic. Frozen on-disk data only; no live DB. PK-BPO carries the full CI (it is the cell
the experiment is about: SE 0.626 vs reranker 0.491). LK-BPO carries the rescore point estimate.
"""
import json, collections, time, subprocess, tempfile, sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/frapercan/Thesis2/storage/deepgose_faithful")
import measure_pk as M

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
PK_POOL = R / "percut_rerank/predictions/pk/pk.tsv"
LK_POOL = R / "percut_rerank/predictions/lk/lk.tsv"
ANCHOR = {"LK-BPO": 0.31323, "PK-BPO": 0.14351}
COMBINE = "avg"

# ---- obo (BP set + alt + ancestors + IA) ----
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

# ---- generic true-frame cafa driver (used for LK, which measure_pk does not cover) ----
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

def load_se(cell):
    Z = np.load(KW / f"se_scores_{cell}.npz", allow_pickle=True)
    prots = Z["proteins"].tolist(); terms = [alt.get(g, g) for g in Z["terms"].tolist()]
    S = Z[COMBINE]
    tpos = {g: j for j, g in enumerate(terms)}
    pidx = {p: i for i, p in enumerate(prots)}
    return prots, terms, S, tpos, pidx

def rank_match_rescore(pool_df, se_lookup, rng, order="se"):
    """Return a new score column: within BP rows that HAVE an SE score, keep the reranker score
    MULTISET but re-assign it in SE order (or random). BP rows without SE and non-BP rows unchanged."""
    new = pool_df["s"].to_numpy(dtype=np.float64).copy()
    bp_has = []
    se_vals = []
    for i, (p, tprim, s) in enumerate(zip(pool_df["p"], pool_df["tprim"], new)):
        if tprim in BP:
            sv = se_lookup.get((p, tprim))
            if sv is not None:
                bp_has.append(i); se_vals.append(sv)
    bp_has = np.array(bp_has); se_vals = np.array(se_vals)
    if len(bp_has) == 0: return new, 0
    ladder = np.sort(new[bp_has])[::-1]           # reranker score multiset, descending
    if order == "se":
        rank = np.argsort(-se_vals, kind="mergesort")   # rows in SE-descending order
    else:
        rank = rng.permutation(len(bp_has))
    new[bp_has[rank]] = ladder
    return new, len(bp_has)

def se_lookup_for(prots, terms, S, pidx, tpos):
    """dict (prot, primary_term) -> se avg, only for cells in the SE matrix (dense build below)."""
    return {"prots": set(prots), "S": S, "pidx": pidx, "tpos": tpos}

def make_lookup(cell):
    prots, terms, S, tpos, pidx = load_se(cell)
    d = {}
    # dense lookup via closures is too slow at 350k rows; build a per-(prot,term) callable
    def get(pt):
        p, g = pt
        i = pidx.get(p); j = tpos.get(g)
        if i is None or j is None: return None
        return float(S[i, j])
    return get, (prots, terms, S, tpos, pidx)

results = {"combine": COMBINE, "model": {
    "name": "clean semantic-entailment ensemble over learned k-WTA codes (deepgose_kwta)",
    "architecture": "DeepGOModel MLP tower (2048 k-WTA d8979601 -> 2560 embed) + ELEmbedding ball "
                    "space (center c_t + radius r_t per GO term), 5 independently-seeded models, "
                    "SE score = ensemble avg of sigmoid(x . (c_t+hasFunc) + r_t)",
    "axioms": "t0 2025-07-22 GO EL normal forms nf1(70618) nf2(10177) nf3(10173) nf4(17699), 8 "
              "relations incl part_of/regulates/has_part; 9655 trainable BP terms + 30251 axiom-only",
    "no_gat": True, "no_mfpreds": True, "no_deepgo2_harness": True,
    "temporal_gate": "labels v227=t0(2025-09-04) propagated BP; blind eval v227->v230; eval-window-new "
                     "BP terms absent from training by construction",
    "corpus": "554,378 curated IEA+EXP proteins; 5 seeds x 15 epochs; valid AUC 0.998"}}

# =========================== PK-BPO (full CI) ===========================
log("===== PK-BPO rescore + CI =====")
gt_fn = str(GTDIR / "groundtruth_PK.tsv"); known = str(GTDIR / "groundtruth_PK_known.tsv")
anchor_cur = M.perprot_curves(str(PK_POOL))
NSBP = "biological_process"
f_a, tau_a, _ = M.fmax_from(anchor_cur[NSBP]["pred_at"], anchor_cur[NSBP]["tp_at"], anchor_cur[NSBP]["n_gt"])
log(f"  anchor PK-BPO reproduced: {round(f_a,5)} (canonical {ANCHOR['PK-BPO']})")
assert abs(f_a - ANCHOR["PK-BPO"]) < 2e-4, f"anchor mismatch {f_a}"

pool = pd.read_csv(PK_POOL, sep="\t", header=None, names=["p", "t", "s"])
pool["tprim"] = pool["t"].map(lambda g: alt.get(g, g))
get_se, _ = make_lookup("PK")
# build fast lookup dict for BP rows only
bp_mask = pool["tprim"].isin(BP).to_numpy()
lut = {}
for p, g in zip(pool["p"].to_numpy()[bp_mask], pool["tprim"].to_numpy()[bp_mask]):
    if (p, g) not in lut:
        v = get_se((p, g))
        if v is not None: lut[(p, g)] = v
log(f"  BP pool rows {int(bp_mask.sum()):,}; with SE score {len(lut):,}")

rng = np.random.default_rng(42)
pk = {"anchor": round(f_a, 5), "tau_anchor": round(tau_a, 2)}
for tag, order in [("B_rescore_SE", "se"), ("R_rescore_random", "random")]:
    new, n_resc = rank_match_rescore(pool, lut, rng, order=order)
    tf = OUT / f"_pk_{tag}.tsv"
    with tf.open("w") as fh:
        for p_, g_, v in zip(pool["p"], pool["t"], new):
            fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
    cur = M.perprot_curves(str(tf))
    f_b, tau_b, _ = M.fmax_from(cur[NSBP]["pred_at"], cur[NSBP]["tp_at"], cur[NSBP]["n_gt"])
    entry = {"f_micro_w": round(f_b, 5), "tau": round(tau_b, 2), "delta": round(f_b - f_a, 5),
             "n_rescored_rows": int(n_resc)}
    if order == "se":
        bs = M.paired_bootstrap(cur, anchor_cur, NSBP, n_boot=1000)
        entry["ci_lo"] = round(bs["ci_lo"], 5); entry["ci_hi"] = round(bs["ci_hi"], 5)
        entry["p_delta_gt0"] = bs["p_gt0"]; entry["n_prot"] = bs["n_prot"]
    pk[tag] = entry
    os.remove(tf)
    log(f"  {tag}: f {entry['f_micro_w']} delta {entry['delta']:+} "
        f"{'CI['+str(entry.get('ci_lo'))+','+str(entry.get('ci_hi'))+'] p>0 '+str(entry.get('p_delta_gt0')) if order=='se' else ''}")

# best generator arm (top5 SE extras added to the pool) + CI
log("  building best generator arm (top5 SE extras) for CI...")
prots, terms, S, tpos, pidx = load_se("PK")
gt = pd.read_csv(gt_fn, sep="\t"); gt = gt[gt.aspect == "P"].copy()
gt["term"] = gt.term.map(lambda g: alt.get(g, g))
gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
targets = [p for p in sorted(gt_leaf) if p in pidx]
pool_bp = pool[pool["tprim"].isin(BP)].groupby("p")["tprim"].apply(set).to_dict()
K = 5
ex = []
for p in targets:
    i = pidx[p]; sc = S[i]; present = pool_bp.get(p, set())
    order = np.argsort(-sc); k = 0
    for j in order:
        g = terms[j]
        if g in present: continue
        ex.append((p, g, float(sc[j]))); k += 1
        if k >= K: break
svals = np.array([e[2] for e in ex]); lo, hi = svals.min(), svals.max()
nrm = lambda v: (v - lo) / (hi - lo) if hi > lo else 0.5
base_rows = list(zip(pool["p"], pool["t"], pool["s"].astype(float)))
gen_rows = base_rows + [(p, g, float(nrm(v))) for (p, g, v) in ex]
tf = OUT / "_pk_generator_top5.tsv"
with tf.open("w") as fh:
    for p_, g_, v in gen_rows: fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
cur = M.perprot_curves(str(tf))
f_g, tau_g, _ = M.fmax_from(cur[NSBP]["pred_at"], cur[NSBP]["tp_at"], cur[NSBP]["n_gt"])
bs = M.paired_bootstrap(cur, anchor_cur, NSBP, n_boot=1000)
pk["G_generator_top5"] = {"f_micro_w": round(f_g, 5), "tau": round(tau_g, 2), "delta": round(f_g - f_a, 5),
    "n_extras": len(ex), "ci_lo": round(bs["ci_lo"], 5), "ci_hi": round(bs["ci_hi"], 5),
    "p_delta_gt0": bs["p_gt0"], "n_prot": bs["n_prot"]}
os.remove(tf)
log(f"  G_generator_top5: f {round(f_g,5)} delta {round(f_g-f_a,5):+} CI[{round(bs['ci_lo'],5)},{round(bs['ci_hi'],5)}] p>0 {bs['p_gt0']}")
results["PK-BPO"] = pk
json.dump(results, open(OUT / "convert_ci.json", "w"), indent=1)

# =========================== LK-BPO (rescore point estimate) ===========================
log("===== LK-BPO rescore (point estimate, no exclusion) =====")
gt_lk = str(GTDIR / "groundtruth_LK.tsv")
poolL = pd.read_csv(LK_POOL, sep="\t", header=None, names=["p", "t", "s"])
poolL["tprim"] = poolL["t"].map(lambda g: alt.get(g, g))
prots, terms, S, tpos, pidx = load_se("LK")
def get_se_lk(pt):
    p, g = pt; i = pidx.get(p); j = tpos.get(g)
    return None if (i is None or j is None) else float(S[i, j])
lutL = {}
bpL = poolL["tprim"].isin(BP).to_numpy()
for p, g in zip(poolL["p"].to_numpy()[bpL], poolL["tprim"].to_numpy()[bpL]):
    if (p, g) not in lutL:
        v = get_se_lk((p, g))
        if v is not None: lutL[(p, g)] = v
log(f"  LK BP pool rows {int(bpL.sum()):,}; with SE {len(lutL):,}")
base_lk = list(zip(poolL["p"], poolL["t"], poolL["s"].astype(float)))
f_al = cafa(base_lk, gt_lk, None)
log(f"  anchor LK-BPO reproduced: {f_al} (canonical {ANCHOR['LK-BPO']})")
rngL = np.random.default_rng(42)
newL, nL = rank_match_rescore(poolL, lutL, rngL, order="se")
rescore_lk = list(zip(poolL["p"], poolL["t"], newL))
f_bl = cafa(rescore_lk, gt_lk, None)
results["LK-BPO"] = {"anchor": f_al, "canonical_anchor": ANCHOR["LK-BPO"],
                     "B_rescore_SE": f_bl, "delta": round(f_bl - f_al, 5) if (f_bl and f_al) else None,
                     "n_rescored_rows": int(nL)}
log(f"  LK-BPO rescore: {f_bl} delta {results['LK-BPO']['delta']:+}" if results['LK-BPO']['delta'] is not None else "  LK failed")
json.dump(results, open(OUT / "convert_ci.json", "w"), indent=1)
log("WROTE convert_ci.json")
print(json.dumps(results, indent=1))
