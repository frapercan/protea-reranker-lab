"""THE ONE-HOUR TEST. Does the set-level prior-knowledge signal add to the reranker
ON THE REAL CANDIDATE POOL, on top of the 72 features we already ship?

Background. On a full-vocabulary task (3,388 candidates/protein, recall 0.94) reading
the KNOWN-TERM SET beat counting marginals by +0.0626 f_micro_w with near-duplicates
excluded, same information to both arms. That is a MECHANISM result on an easy candidate
regime, not a product number: the deployed pool is ~140 candidates at recall 0.322. This
asks the only question that decides whether to build anything: does it survive as ONE
FEATURE on the real pool, next to everything else we already carry?

DESIGN, and its one real limitation. The reranker's own train.parquet holds proteins from
earlier snapshot pairs whose t0-known terms are NOT frozen anywhere (the benchmark freezes
knowledge only for the eval targets). So the feature cannot be computed for those rows.
The A/B therefore runs INSIDE the eval cell: split the 4,455 PK-BPO proteins 70/30, train
on one part, test on the other. Both arms see identical rows and identical features; the
only difference is the added column. That makes this a valid A/B for "does the feature
add", and NOT a reproduction of the deployed recipe (which trains on train.parquet).

  A  the 72 features we ship                     control
  B  the 72 + set_level_transfer                 the question

LEAKAGE, stated plainly. The transfer source is the reranker-train proteins' BP answers,
which are post-t0; production would transfer from a t0 corpus. So B is OPTIMISTIC by
construction. Two guards:
  * near-duplicate neighbours (Jaccard >= MAXJ) are EXCLUDED from the transfer, so no
    test protein can be answered by copying a twin. This is what took the full-vocab gain
    from +0.2125 down to its honest +0.0626.
  * the split is by PROTEIN, so no protein informs itself.
Read a positive here as "worth an export to test properly on LK", not as a board delta.

AUC is deliberately NOT the criterion. On this exact question AUC has now ordered the
arms BACKWARDS four times: counting carries a better AUC (0.9393 vs 0.8569) and half the
f_micro_w. f_micro_w on the same harness decides.
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
MAXJ = 0.5
SEED = 42
EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
FEATS = [c for c in pq.ParquetFile(DS / "eval.parquet").schema_arrow.names if c not in EX]

# ---- the real pool: eval.parquet PK-BPO rows ---------------------------------
cols = FEATS + ["category", "aspect", "protein_accession", "go_term_id", "label", "snapshot_pair"]
t = pq.read_table(DS / "eval.parquet", columns=list(dict.fromkeys(cols)))
cat = np.asarray(t.column("category").to_pylist())
asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float32)
for j, c in enumerate(FEATS):
    X[:, j] = t.column(c).to_numpy(zero_copy_only=False).astype(np.float32)[m]
prot = np.asarray(t.column("protein_accession").to_pylist())[m]
go = np.asarray(t.column("go_term_id").to_pylist())[m]
lab = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.int32)
print(f"real PK-BPO pool: {X.shape[0]:,} rows over {len(set(prot)):,} proteins, "
      f"{X.shape[1]} features, pos_rate={lab.mean():.4f}", flush=True)

# ---- frozen t0 knowledge + the answers ---------------------------------------
K = {}
for i, l in enumerate(open(REL / "groundtruth_PK_known.tsv")):
    if i == 0:
        continue
    p = l.rstrip("\n").split("\t")
    if len(p) >= 2:
        K.setdefault(p[0], set()).add(p[1])
TRUTH = [tuple(l.rstrip("\n").split("\t")) for l in open(DS / "gt_pk_bp.tsv")]
BP = collections.defaultdict(set)
for p_, g_ in TRUTH:
    BP[p_].add(g_)

ps = sorted(set(prot))
rng = np.random.default_rng(SEED)
rng.shuffle(ps)
cut = int(0.7 * len(ps))
TRp, TEp = set(ps[:cut]), set(ps[cut:])
print(f"split by protein: train {len(TRp):,} / test {len(TEp):,}  "
      f"(with frozen t0 knowledge: {sum(1 for p in ps if p in K):,})", flush=True)

# ---- the new feature: set-level transfer, near-duplicates excluded ------------
src = [p for p in TRp if p in K and BP.get(p)]
src_sets = [K[p] for p in src]
inv = collections.defaultdict(list)
for j, s in enumerate(src_sets):
    for k in s:
        inv[k].append(j)


def transfer(kq, topn=30):
    hits = collections.Counter()
    for k in kq:
        for j in inv.get(k, ()):
            hits[j] += 1
    sims = []
    for j, inter in hits.items():
        u = len(kq) + len(src_sets[j]) - inter
        if not u:
            continue
        jac = inter / u
        if jac < MAXJ:                      # never copy a twin
            sims.append((jac, j))
    if not sims:
        return {}
    sims.sort(reverse=True)
    sims = sims[:topn]
    tot = sum(s for s, _ in sims) or 1.0
    out = collections.Counter()
    for s, j in sims:
        for tt in BP[src[j]]:
            out[tt] += s
    return {tt: v / tot for tt, v in out.items()}


t0 = time.time()
cache = {}
new = np.zeros(len(prot), dtype=np.float32)
for i, (p, g) in enumerate(zip(prot, go)):
    if p not in cache:
        cache[p] = transfer(K.get(p, set()))
    new[i] = cache[p].get(g, 0.0)
print(f"feature built for {len(cache):,} proteins; nonzero on {(new > 0).mean():.1%} of rows "
      f"({time.time()-t0:.0f}s)", flush=True)

itr = np.array([p in TRp for p in prot])
ite = ~itr
# a small inner validation split for early stopping, carved from train proteins
vp = set(list(TRp)[: max(1, len(TRp) // 6)])
iv = np.array([p in vp for p in prot]) & itr
ifit = itr & ~iv

P = {"objective": "lambdarank", "metric": ["ndcg"], "ndcg_eval_at": [5, 10], "label_gain": [0, 1],
     "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.9,
     "bagging_fraction": 0.9, "bagging_freq": 5, "seed": SEED, "verbose": -1, "num_threads": 12,
     "max_bin": 63, "two_round": True, "force_col_wise": True}


def groups(a):
    o = np.argsort(a, kind="stable")
    _, c = np.unique(a[o], return_counts=True)
    return o, c


def run(name, Xm, names):
    of, cf = groups(prot[ifit])
    ov, cv = groups(prot[iv])
    dtr = lgb.Dataset(Xm[ifit][of], label=lab[ifit][of], group=cf, feature_name=names)
    dva = lgb.Dataset(Xm[iv][ov], label=lab[iv][ov], group=cv, reference=dtr, feature_name=names)
    b = lgb.train(P, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    s = b.predict(Xm[ite], num_iteration=b.best_iteration)
    imp = None
    if "set_level_transfer" in names:
        gains = b.feature_importance("gain")
        k = names.index("set_level_transfer")
        imp = round(float(gains[k] / max(gains.sum(), 1) * 100), 2)
    DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / f"{name}.tsv").open("w") as fh:
            for p_, g_, v in zip(prot[ite], go[ite], s):
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, g_ in TRUTH:
                if p_ in TEp:                      # score only the held-out proteins
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None, imp, b.best_iteration
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return ({k: round(float(best[k]), 4) for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")}
                if best else None), imp, b.best_iteration


res = {"note": "A/B inside the eval cell on the REAL pool. Identical rows; the only "
               "difference is the added column. Transfer excludes near-duplicates "
               f"(Jaccard >= {MAXJ}). Optimistic by construction: the transfer source is "
               "post-t0. Read as go/no-go for a proper LK export, not as a board delta.",
       "rows": int(X.shape[0]), "test_proteins": len(TEp), "pos_rate": round(float(lab.mean()), 4),
       "feature_nonzero_frac": round(float((new > 0).mean()), 4)}

a, _, ai = run("A", X, FEATS)
res["A_shipped_72_features"] = a
res["A_best_iter"] = ai
print(f"[A] the 72 we ship        {a}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "set_level_as_reranker_feature.json", "w"), indent=1)

XB = np.hstack([X, new.reshape(-1, 1)])
b_, bimp, bi = run("B", XB, FEATS + ["set_level_transfer"])
res["B_plus_set_level"] = b_
res["B_best_iter"] = bi
res["set_level_gain_share_pct"] = bimp
print(f"[B] + set_level_transfer  {b_}   (feature gain share: {bimp}%)  ({time.time()-t0:.0f}s)", flush=True)

fa = (a or {}).get("f_micro_w")
fb = (b_ or {}).get("f_micro_w")
if fa and fb:
    res["delta"] = round(fb - fa, 4)
json.dump(res, open(W / "set_level_as_reranker_feature.json", "w"), indent=1)
print("\n=== Does the set-level channel survive on the REAL pool? ===", flush=True)
print(f"  A the 72 features we ship = {fa}", flush=True)
print(f"  B + set_level_transfer    = {fb}", flush=True)
if fa and fb:
    print(f"  -> delta = {fb-fa:+.4f}", flush=True)
    print("  GO if clearly positive: then export t0 for LK and test the cell that needs it.", flush=True)
    print("  NO-GO if flat: the full-vocab +0.0626 was the easy candidate regime. Close the line.", flush=True)
print("DONE", flush=True)
