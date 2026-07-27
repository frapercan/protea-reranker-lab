"""PHYLO STEP 2 -- does unioning the phylo-profile proposals into the deployed pool raise f_micro_w
in the TRUE board frame, vs the DEPLOYED percutgraft anchor (PK 0.14351 / LK 0.31323)?

ONE VARIABLE: candidate set only.
  A  = deployed percut_rerank prediction file, VERBATIM (the anchor pool).
  B  = A + phylo BP extras (top-k per protein not already in pool), scored by min-max-normalised
       phylo confidence (the generator's own score; DB-free analog of the raw-sigmoid arm).
CONTROLS (re-check every positive):
  U  = A + matched-volume UNIFORM extras: same COUNT of extras per protein but terms drawn at the
       deployed constant score from the protein's own phylo candidate vocabulary in random order
       (isolates candidate SET/volume from ORDER).
  Rnd= A + phylo extras with SHUFFLED scores (same set, random order/scale).

TRUE frame: lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds -known (evaluation.nf:279).
cafa_eval via repositories/PROTEA/.venv/bin/python. Generator identical to phylo_gate.py.
"""
import json, collections, time, subprocess, tempfile, random
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
PREDDIR = R / "percut_rerank/predictions"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
PH = ROOT / "storage/phylo_profile"
OUT = ROOT / "storage/regen_headline"
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
TOPN_NEIGH, SIM_MIN, MIN_PREV, MAX_PREV_FRAC = 50, 0.20, 4, 0.95
EXTRA_KS = [5, 10, 25]
ANCHOR = {"LK-BPO": 0.31323, "PK-BPO": 0.14351}
random.seed(42); np.random.seed(42)


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


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
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stack.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r
log(f"obo {len(BP):,} BP")

meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id)); id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
acc2bpraw = ref.dropna(subset=["go"]).query("asp=='P'").groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
_bpprop = {}
def bp_prop(acc):
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc)
        r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset()
        _bpprop[acc] = r
    return r

# phylo profiles
Pmat = sparse.load_npz(PH / "og_presence.npz").tocsr()
mm = np.load(PH / "og_meta.npz", allow_pickle=True)
ogs = list(mm["ogs"]); prevalence = mm["prevalence"]; ntax = Pmat.shape[1]
og_idx = {o: i for i, o in enumerate(ogs)}
acc2og = json.load(open(PH / "acc2og.json"))
og_members = json.load(open(PH / "og_members.json"))
D = Pmat.toarray().astype(np.float32); mu = D.mean(axis=1, keepdims=True); Dc = D - mu
norm = np.sqrt((Dc * Dc).sum(axis=1, keepdims=True)); norm[norm == 0] = 1.0
Dn = Dc / norm; del D, Dc
informative = (prevalence >= MIN_PREV) & (prevalence <= MAX_PREV_FRAC * ntax)
cand_idx = np.where(informative)[0]; cand_norm = Dn[cand_idx]
og_bp = {}
for og, accs in og_members.items():
    s = frozenset().union(*[bp_prop(a) for a in accs]) if accs else frozenset()
    if s: og_bp[og] = s
log(f"profiles {Pmat.shape}; informative {len(cand_idx):,}; OGs w/BP {len(og_bp):,}")


def neighbours(xi, exclude):
    best = {}
    for i in xi:
        v = Dn[i]
        if not np.any(v): continue
        sims = cand_norm @ v
        for j in np.argsort(-sims)[:TOPN_NEIGH * 3]:
            s = float(sims[j])
            if s < SIM_MIN: break
            og = ogs[cand_idx[j]]
            if og in exclude: continue
            if s > best.get(og, -1): best[og] = s
    return sorted(best.items(), key=lambda kv: -kv[1])[:TOPN_NEIGH]


def phylo_scores(targets):
    score = {}
    for p in targets:
        ogx = acc2og.get(p, [])
        if not ogx: continue
        xi = [og_idx[o] for o in ogx if o in og_idx]
        nb = neighbours(xi, set(ogx))
        sd = collections.defaultdict(float)
        for og_y, s in nb:
            accs = og_members.get(og_y, ())
            By = og_bp.get(og_y)
            if p in accs:
                By = frozenset().union(*[bp_prop(a) for a in accs if a != p]) or frozenset()
            if not By: continue
            for t in By: sd[t] += s
        if sd: score[p] = sd
    return score


DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs = cafa_eval("{obo}", "{pd}", "{gt}", ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, toi_file="{toi}", exclude={known}, max_terms=None, th_step=0.01, n_cpu=6, weighted_only=False)
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
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=gt_file, ia=IA_F, toi=TOI, known=knownrepr, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None: best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


results = {}
for cell, cat, gt_fn, known in [
    ("LK-BPO", "lk", str(GTDIR / "groundtruth_LK.tsv"), None),
    ("PK-BPO", "pk", str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv")),
]:
    log(f"===== {cell} =====")
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None, names=["prot", "term", "score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    base = list(dep.itertuples(index=False, name=None))
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    targets = sorted(pd.read_csv(gt_fn, sep="\t").query("aspect=='P'").EntryID.unique())
    tb = tb[tb.protein_accession.isin(set(targets))]; tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    pool_bp = tb[tb.go.isin(BP)].groupby("protein_accession").go.apply(set).to_dict()

    scores = phylo_scores(targets)
    allv = [v for p in scores for v in scores[p].values()]
    lo, hi = (min(allv), max(allv)) if allv else (0.0, 1.0)
    def nrm(v): return (v - lo) / (hi - lo) if hi > lo else 0.5
    log(f"  targets {len(targets)}; deployed rows {len(base):,}; phylo-scored {len(scores):,}")

    fa = cafa(base, gt_fn, known)
    log(f"  A anchor (TRUE frame{' -known' if known else ''}): {fa}  (canonical {ANCHOR[cell]})")
    cell_res = {"A_pool_only_trueframe": fa, "canonical_anchor": ANCHOR[cell], "extras": {}}
    for K in EXTRA_KS:
        ex, uni, rnd = [], [], []
        for p in targets:
            sd = scores.get(p)
            if not sd: continue
            present = pool_bp.get(p, set())
            ranked = [(g, sv) for g, sv in sorted(sd.items(), key=lambda kv: -kv[1]) if sv > 0 and g not in present]
            picks = ranked[:K]
            for g, sv in picks: ex.append((p, g, nrm(sv)))
            # matched-volume UNIFORM: same count, terms from protein's own candidate vocab, random order, const score
            vocab = [g for g, _ in ranked]
            random.shuffle(vocab)
            for g in vocab[:len(picks)]: uni.append((p, g, 0.5))
            # random-order/scale: same set, shuffled scores
            shufv = [nrm(sv) for _, sv in picks]; random.shuffle(shufv)
            for (g, _), sv in zip(picks, shufv): rnd.append((p, g, sv))
        fb = cafa(base + ex, gt_fn, known)
        fu = cafa(base + uni, gt_fn, known)
        fr = cafa(base + rnd, gt_fn, known)
        cell_res["extras"][f"top{K}"] = {
            "n_extras": len(ex),
            "B_phylo": fb, "delta_B_minus_A": round(fb - fa, 5) if fb and fa else None,
            "U_matched_uniform": fu, "delta_U_minus_A": round(fu - fa, 5) if fu and fa else None,
            "Rnd_shuffled": fr, "delta_Rnd_minus_A": round(fr - fa, 5) if fr and fa else None,
        }
        d = cell_res["extras"][f"top{K}"]
        log(f"  top{K}: +{len(ex):,} extras  B {fb} (d {d['delta_B_minus_A']:+}) | U {fu} (d {d['delta_U_minus_A']:+}) | Rnd {fr} (d {d['delta_Rnd_minus_A']:+})")
        results[cell] = cell_res
        json.dump(results, open(OUT / "phylo_step2_cafaeval.json", "w"), indent=2)
    results[cell] = cell_res
    json.dump(results, open(OUT / "phylo_step2_cafaeval.json", "w"), indent=2)

json.dump(results, open(OUT / "phylo_step2_cafaeval.json", "w"), indent=2)
log("wrote phylo_step2_cafaeval.json")
print(json.dumps(results, indent=2))
