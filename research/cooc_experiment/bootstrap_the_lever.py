"""The interval the lever needs to be publishable, and the artefact it needs to be reusable.

WHAT IS ESTABLISHED. `score_the_extras.py` found the peak and the shape:

  top-k   extras     B = A + extras scored by S      C = A + extras at the raw sigmoid
  5       1,803      0.22922  (+0.0064)              0.22796  (+0.0051)
  20      16,445     0.24250  (+0.0197)              0.23420  (+0.0114)
  **50**  76,348     **0.24527 (+0.02245)**          0.22262  (-0.0002)
  100     226,553    0.23688  (+0.0141)              0.21875  (-0.0041)
  200     590,117    0.23224  (+0.0094)              0.22004  (-0.0028)

The peak is **top50** and the curve rises and falls exactly as `prop=fill`'s tax predicts: every
submitted false extra forfeits its ancestor's free inheritance, so a better score does not remove
the tax, it moves the depth at which the tax catches you (top20 for the raw sigmoid, top50 for S).
At top50 the SAME 76,348 candidates LOSE with the sigmoid and WIN by +0.0225 with S: the difference
is the scoring and nothing else.

WHAT IS MISSING, and it is the difference between a result and a number. There is ONE temporal
split and NO interval. +0.02245 clears the 0.0034 fold-noise floor by six times and the
dose-response is monotone in both directions, which are the two things that make a number credible,
and **neither is a confidence interval**. Nine tidy stories have died in this campaign and several
had better-looking evidence than this.

THE MEASUREMENT. Resample the blind window's proteins with replacement, and score arm A and arm B on
**the same resample**, so the difference is paired and the protein-to-protein variance cancels where
it should. Report the 2.5/97.5 percentiles of (B - A).

THE GATE, quantity named: the 95% interval of (B - A) must **exclude zero**, and its lower bound must
clear **0.0034**. An interval that spans zero means the lever is one temporal split's luck.

THE ARTEFACT. This also writes `lever_submission_top50.tsv`: the exact rows arm B submits, so the
next run does not repeat seven minutes of frame building and training, and so the number has a file
behind it. That is the standing rule of this campaign and this lever does not get an exemption:
`predictions_protea.tsv` is the cautionary tale, a headline with no artefact.

THE CAVEAT THAT DOES NOT GO AWAY. The generator is `classifier_6plm_asl`, which the author correctly
calls old: 2026-06-14, six RAW PLMs rather than the learned k-WTA champion, no provenance beyond
`mu`/`sd`/`vocab`, architecture recovered from tensor shapes. Whatever interval this produces, it is
an interval around a FLOOR.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np, yaml, psycopg2, pyarrow.parquet as pq, torch, torch.nn as nn, lightgbm as lgb

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED, VAL_PAIR, TAU, TOPK, NBOOT = 42, "v225-v227", 0.393, 50, 20
ANCHOR = 0.22288
ORDER = [("08234f06-ba76-4d7d-aaec-ae601096b4fa", 768), ("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560),
         ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536), ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280),
         ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152), ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass
par = collections.defaultdict(set); ns = {}; alt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("namespace: "):
        ns[cur_] = line[11:]
    elif cur_ and line.startswith("is_a: GO:"):
        par[cur_].add(line[6:].split(" ! ")[0].strip())
    elif cur_ and line.startswith("relationship: part_of GO:"):
        par[cur_].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur_ and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur_
BP = {t for t, n in ns.items() if n == "biological_process"}

cfg = yaml.safe_load(Path("/home/frapercan/Thesis2/repositories/PROTEA/protea/config/system.yaml").read_text())
url = None
def find(d):
    global url
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, str) and v.startswith("postgresql"):
                url = v
            else:
                find(v)
    elif isinstance(d, list):
        for v in d:
            find(v)
find(cfg)
url = url.replace("postgresql+psycopg://", "postgresql://")


def frame(accs):
    conn = psycopg2.connect(url); conn.set_session(readonly=True)
    cur = conn.cursor()
    X = np.zeros((len(accs), 8320), np.float32)
    pos = {a: i for i, a in enumerate(accs)}
    off = 0
    for cid, dim in ORDER:
        for s in range(0, len(accs), 4000):
            ch = accs[s:s + 4000]
            cur.execute("""select p.accession, e.embedding::text from protein p
                           join sequence_embedding e on e.sequence_id = p.sequence_id
                           where e.embedding_config_id = %s and p.accession = any(%s)""", (cid, ch))
            for acc, emb in cur.fetchall():
                v = np.fromstring(emb.strip("[]"), sep=",", dtype=np.float32)
                if v.shape[0] == dim and acc in pos:
                    X[pos[acc], off:off + dim] = v
        off += dim
    conn.close()
    return X


t0 = time.time()
CACHE = W / "lever_submission_top50.tsv"
GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(alt.get(t_, t_))

if CACHE.exists():
    print(f"reusing {CACHE.name}: the submission is already on disk, no rebuild", flush=True)
    rows = []
    for line in CACHE.open():
        p_, g_, v, tag = line.rstrip("\n").split("\t")
        rows.append((p_, g_, float(v), tag))
else:
    ck = torch.load("/home/frapercan/Thesis2/storage/fullgo_models/classifier_6plm_asl.pt",
                    map_location="cpu", weights_only=False)
    vocab = [alt.get(g, g) for g in ck["vocab"]]
    vpos = {g: i for i, g in enumerate(vocab)}
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    net = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(1024, len(vocab)))
    miss, _ = net.load_state_dict({k[4:] if k.startswith("net.") else k: v
                                   for k, v in ck["state_dict"].items()}, strict=False)
    assert not miss, f"partially loaded model: {miss}"
    net = net.to(DEV).eval()
    mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
    GVn = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
    GVc = GVn["codes"].astype(np.float32)
    gpos = {alt.get(g, g): i for i, g in enumerate(GVn["go_ids"].tolist())}
    az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
    AVc = np.vstack([az["embeddings"].astype(np.float32), np.zeros((1, az["embeddings"].shape[1]), np.float32)])
    apos = {alt.get(g, g): i for i, g in enumerate(az["go_ids"].tolist())}
    NA = len(AVc) - 1
    QV = np.load(W / "d8979601_full" / "codes.npy")
    qpos = {a: i for i, a in enumerate(json.load(open(W / "d8979601_full" / "accs.json")))}
    QVz = np.vstack([QV, np.zeros((1, QV.shape[1]), np.float32)])
    GVz = np.vstack([GVc, np.zeros((1, GVc.shape[1]), np.float32)])

    def feats(P, G, LOGm, fposm):
        fi = np.array([fposm.get(p, -1) for p in P]); gi = np.array([vpos.get(g, -1) for g in G])
        ok = (fi >= 0) & (gi >= 0)
        logit = np.zeros(len(P), np.float32)
        logit[ok] = LOGm[fi[ok], gi[ok]]
        return np.hstack([logit[:, None],
                          QVz[np.array([qpos.get(p, len(QV)) for p in P])],
                          GVz[np.array([gpos.get(g, len(GVc)) for g in G])],
                          AVc[np.array([apos.get(g, NA) for g in G])],
                          np.array([IA.get(g, 0.0) for g in G], np.float32)[:, None],
                          np.array([len(par.get(g, ())) for g in G], np.float32)[:, None]]).astype(np.float32)

    def rows_of(path, part=None):
        have = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in ("protein_accession", "go_term_id", "category", "aspect", "label",
                            "snapshot_pair") if c in have]
        t = pq.read_table(path, columns=cols)
        C = np.asarray(t.column("category").to_pylist()); A = np.asarray(t.column("aspect").to_pylist())
        m = (C == "pk") & (A == "bpo")
        P = np.asarray(t.column("protein_accession").to_pylist())[m]
        G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
        Y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.float32)
        S = (np.asarray(t.column("snapshot_pair").to_pylist())[m] if "snapshot_pair" in cols
             else np.full(len(P), "eval"))
        k = (S != VAL_PAIR) if part == "past" else (S == VAL_PAIR) if part == "val" else np.ones(len(P), bool)
        return P[k], G[k], Y[k]

    def subsample(P, G, Y, ratio=5, seed=SEED):
        pos = np.where(Y > 0)[0]; neg = np.where(Y == 0)[0]
        take = np.random.default_rng(seed).choice(neg, min(len(pos) * ratio, len(neg)), replace=False)
        idx = np.concatenate([pos, take])
        return P[idx], G[idx], Y[idx]

    trP, trG, trY = subsample(*rows_of(DS / "train.parquet", "past"))
    vaP, vaG, vaY = subsample(*rows_of(DS / "train.parquet", "val"))
    need = sorted(set(trP.tolist()) | set(vaP.tolist()))
    Xf = frame(need); fpos = {a: i for i, a in enumerate(need)}
    Xn = torch.tensor((Xf - mu) / sd, dtype=torch.float32); del Xf
    with torch.no_grad():
        LOG = np.vstack([net(Xn[i:i + 256].to(DEV)).cpu().numpy() for i in range(0, len(Xn), 256)])
    del Xn
    print(f"  train logits {LOG.shape}  ({time.time()-t0:.0f}s)", flush=True)
    _P = float(trY.sum()); _N = float(len(trY) - _P)
    S_ = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
                    "min_data_in_leaf": 100, "feature_fraction": 0.5, "bagging_fraction": 0.8,
                    "bagging_freq": 5, "seed": SEED, "verbose": -1, "num_threads": 12,
                    "max_bin": 63, "scale_pos_weight": _N / _P, "metric": "average_precision"},
                   lgb.Dataset(feats(trP, trG, LOG, fpos), label=trY), num_boost_round=1500,
                   valid_sets=[lgb.Dataset(feats(vaP, vaG, LOG, fpos), label=vaY)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    del LOG
    print(f"  S trained, best_iteration {S_.best_iteration}  ({time.time()-t0:.0f}s)", flush=True)

    t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
    c = np.asarray(t.column("category").to_pylist()); a = np.asarray(t.column("aspect").to_pylist())
    m = (c == "pk") & (a == "bpo")
    eP = np.asarray(t.column("protein_accession").to_pylist())[m]
    eG = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
    eS = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
    pool = collections.defaultdict(dict)
    for p_, g_, s_ in zip(eP, eG, eS):
        pool[p_][g_] = s_
    eprots = sorted(set(GT) & set(pool))
    Xe = frame(eprots); epos = {a_: i for i, a_ in enumerate(eprots)}
    Xen = torch.tensor((Xe - mu) / sd, dtype=torch.float32); del Xe
    with torch.no_grad():
        ELOG = np.vstack([net(Xen[i:i + 256].to(DEV)).cpu().numpy() for i in range(0, len(Xen), 256)])
    del Xen
    vb = [i for i, g in enumerate(vocab) if g in BP]
    vbg = [vocab[i] for i in vb]
    ex_p, ex_g = [], []
    for i, prot in enumerate(eprots):
        pl = pool[prot]
        top = np.argpartition(-ELOG[i][vb], min(TOPK, len(vb) - 1))[:TOPK]
        for j in top:
            g = vbg[j]
            if g not in pl:
                ex_p.append(prot); ex_g.append(g)
    ex_p, ex_g = np.array(ex_p), np.array(ex_g)
    sc = S_.predict(feats(ex_p, ex_g, ELOG, epos), num_iteration=S_.best_iteration)
    thr = np.quantile(sc, 0.5)
    rows = [(p_, g_, float(s_), "pool") for p_, g_, s_ in zip(eP, eG, eS) if s_ > TAU]
    rows += [(p_, g_, float(v), "extra") for p_, g_, v in zip(ex_p, ex_g, sc) if v > thr]
    with CACHE.open("w") as fh:
        for p_, g_, v, tag in rows:
            fh.write(f"{p_}\t{g_}\t{v:.6f}\t{tag}\n")
    print(f"  wrote {CACHE.name}: {len(rows):,} rows "
          f"({sum(1 for r in rows if r[3]=='extra'):,} extras)  ({time.time()-t0:.0f}s)", flush=True)

base = [(p, g, v) for p, g, v, tag in rows if tag == "pool"]
allr = [(p, g, v) for p, g, v, _ in rows]
print(f"arm A: {len(base):,} pool rows | arm B: {len(allr):,} rows  ({time.time()-t0:.0f}s)", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def cafa(rws, prots=None):
    keep = rws if prots is None else [r for r in rws if r[0] in prots]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in keep:
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_ in (prots if prots is not None else GT):
                for g_ in GT.get(p_, ()):
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA_F, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


fa, fb = cafa(base), cafa(allr)
res = {"topk": TOPK, "anchor_prefiltered": ANCHOR, "A": fa, "B": fb,
       "B_minus_A": round(fb - fa, 5) if (fa and fb) else None,
       "PRECONDITION": bool(fa is not None and abs(fa - ANCHOR) < 0.002),
       "gate": "the 95% interval of (B-A) must exclude zero AND its lower bound must clear 0.0034",
       "caveat": "the generator is classifier_6plm_asl: 2026-06-14, six RAW PLMs, not the learned "
                 "k-WTA champion, no provenance. This is an interval around a FLOOR."}
print(f"\n  A {fa} (anchor {ANCHOR}, precondition {res['PRECONDITION']}) | B {fb} | B-A {res['B_minus_A']}", flush=True)
if not res["PRECONDITION"]:
    print("  PRECONDITION FAILED. VOID.", flush=True)
    json.dump(res, open(W / "bootstrap_the_lever.json", "w"), indent=1)
    raise SystemExit(0)

prots = sorted({p for p, _, _ in allr} & set(GT))
rng = np.random.default_rng(SEED)
boot = []
for i in range(NBOOT):
    samp = set(rng.choice(prots, size=len(prots), replace=True).tolist())
    xa, xb = cafa(base, samp), cafa(allr, samp)        # the SAME resample for both arms: paired
    if xa is not None and xb is not None:
        boot.append(round(xb - xa, 5))
        print(f"    bootstrap {len(boot):2d}: B-A = {boot[-1]:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({**res, "bootstrap_so_far": boot}, open(W / "bootstrap_the_lever.json", "w"), indent=1)
lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
res["bootstrap"] = {"n": len(boot), "deltas": boot, "mean": round(float(np.mean(boot)), 5),
                    "ci95": [round(lo, 5), round(hi, 5)], "excludes_zero": bool(lo > 0),
                    "lower_bound_clears_the_floor": bool(lo > 0.0034)}
res["LEVER_HOLDS"] = bool(res["PRECONDITION"] and lo > 0.0034)
json.dump(res, open(W / "bootstrap_the_lever.json", "w"), indent=1)
print(f"\n=== the lever, with an interval ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  A (pool, prefiltered)  {fa}", flush=True)
print(f"  B (+ extras scored)    {fb}", flush=True)
print(f"  B - A                  {res['B_minus_A']}", flush=True)
print(f"  bootstrap 95% CI       [{lo:.5f}, {hi:.5f}]  over {len(boot)} paired resamples", flush=True)
print(f"  excludes zero: {res['bootstrap']['excludes_zero']} | lower bound clears 0.0034: "
      f"{res['bootstrap']['lower_bound_clears_the_floor']}", flush=True)
print(f"  -> LEVER HOLDS: {res['LEVER_HOLDS']}", flush=True)
print("DONE", flush=True)
