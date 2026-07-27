"""DOES-IT-PAY: does the SEMANTIC-abstract candidate generator pay in the cafaeval frame, the way the
classifier's did? Same scorer S, same anchor, same cafaeval legacy frame as score_the_extras.py; the
ONLY variable is where the extra candidates come from. Four arms:
  A            the pool alone at TAU=0.393 (anchor 0.22288)
  B_classifier pool + the classifier's top-k BP extras scored by S  (reproduces the +0.02245 lever)
  B_semantic   pool + the semantic encoder's top-k BP extras scored by S  (the text channel)
  B_union      pool + (classifier UNION semantic) extras scored by S  (are they complementary?)
Semantic extras = for each eval protein, top-k BP-with-def by MAX cosine (S-PubMedBert) between its
pre-t0 abstracts and the term DEFINITION, filtered to NOT already in the pool. If B_union >
B_classifier the two generators find different missing branches and the path to the two lost cells
runs through generation from BOTH. Everything below (scorer S, anchor, feats, cafa) is unchanged.

WHERE THE SCORER COMES FROM. `fullgo_ceiling.py` measured what the old, unoptimised, raw-PLM full-GO
classifier proposes for PK-BP. Its candidates are true **11.59%** of the time at top5 and **9.84%**
at top20. The bar was 1.85% (the co-occurrence expansion the thesis kills) and 0.19% (descending the
DAG, which collapsed the system to 0.0218). **This generator knows where the missing branches are.**

But the real arm did not win: submitting the extras with the classifier's own sigmoid gave 0.21563
at top20 against the deployed 0.21131, **+0.0025**, under the 0.0034 noise floor, and top50 lost
outright. The generation works. The scoring does not.

So: score them properly.

THE FEATURE PROBLEM, and it is the same one PKW.1 found. The extras have **no kNN features**. The 72
columns come from the neighbour path, which never ran over candidates the classifier proposed. So a
scorer over the union can only use what exists for **any** (protein, term) pair:

  the full-GO classifier's own logit for that pair
  the protein vector      `d8979601`, 2048-d, the deployed encoder
  the GO term code        two-tower sparse codes, 1024-d, fit on v227 = our t0
  the term's anc2vec      200-d, the 2020-10 release
  the term's information accretion and its t0 pool frequency

**Every one of those is defined for a pair the kNN never saw**, which is the whole point.

THE DESIGN, and the transfer is the load-bearing part:
  train S on the POOL rows of the temporal past (v160..v225), which are the only rows that carry a
  label, using ONLY the features above. Early-stop on v225-v227, as deployed. Then apply S to the
  classifier's extra candidates on the blind window. The model never sees a kNN feature, so nothing
  it learned is unavailable at the moment it has to judge an extra.

  A  the pool alone, at the deployed TAU=0.393                  the anchor
  B  A, plus the extras scored by S                             the question
  C  A, plus the extras scored by the classifier's raw sigmoid  the arm `fullgo_ceiling` already ran

THE PRECONDITION: arm A must reproduce **0.22288** within 0.002. TAU=0.393 makes A the PREFILTERED
arm, which `build_submission.py` measures at 0.22288 on the board's line. The first run of this
script anchored on 0.2131, which is the `guard_only` arm that submits everything above zero, got
0.22282, and voided itself. **The precondition was right and my constant was wrong**, and 0.22282
against 0.22288 is an independent reproduction of PKW.8 to four decimals.

THE GATE, quantity named: (B - A) on the blind window must clear **0.0034**, this cell's established
fold noise, and beat C, which is +0.0025 and already inside the noise. If S cannot beat the raw
sigmoid, the scoring is not the bottleneck and the honest conclusion is that this generator's
precision does not survive contact with the metric.

THE MECHANISM IS AGAINST US, as always: under `prop=fill` every false extra we submit forfeits its
ancestor's free inheritance. That is why C peaks at top20 and loses at top50. A better score does not
remove that tax; it only decides which candidates pay it.

NOTE ON THE MODEL, which is the author's own caveat and it stands: `classifier_6plm_asl` is from
2026-06-14, uses six RAW PLMs and not the learned k-WTA champion, records no provenance beyond
`mu`/`sd`/`vocab`, and its architecture had to be recovered from tensor shapes. Whatever this run
says, it says about a floor.
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
SEED, VAL_PAIR, TAU = 42, "v225-v227", 0.393
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
    """The classifier's 8320-d input, read-only from the DB, in the canonical PLM order."""
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


ck = torch.load("/home/frapercan/Thesis2/storage/fullgo_models/classifier_6plm_asl.pt",
                map_location="cpu", weights_only=False)
vocab = [alt.get(g, g) for g in ck["vocab"]]
mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
net = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, len(vocab)))
miss, unexp = net.load_state_dict({k[4:] if k.startswith("net.") else k: v
                                   for k, v in ck["state_dict"].items()}, strict=False)
assert not miss, f"partially loaded model: {miss}"
net = net.to(DEV).eval()
vpos = {g: i for i, g in enumerate(vocab)}
print(f"classifier loaded: 8320 -> {len(vocab):,} labels, every parameter matched", flush=True)

GV = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
GVc = GV["codes"].astype(np.float32)
gpos = {alt.get(g, g): i for i, g in enumerate(GV["go_ids"].tolist())}
az = np.load("/home/frapercan/Thesis2/repositories/PROTEA/artifacts/anc2vec/anc2vec_2020-10.npz", allow_pickle=True)
AVc = np.vstack([az["embeddings"].astype(np.float32), np.zeros((1, az["embeddings"].shape[1]), np.float32)])
apos = {alt.get(g, g): i for i, g in enumerate(az["go_ids"].tolist())}
NA = len(AVc) - 1
QV = np.load(W / "d8979601_full" / "codes.npy")
qpos = {a: i for i, a in enumerate(json.load(open(W / "d8979601_full" / "accs.json")))}

t0 = time.time()


def rows_of(path, part=None):
    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in ("protein_accession", "go_term_id", "category", "aspect", "label",
                        "snapshot_pair", "go_term_frequency") if c in have]
    t = pq.read_table(path, columns=cols)
    C = np.asarray(t.column("category").to_pylist()); A = np.asarray(t.column("aspect").to_pylist())
    m = (C == "pk") & (A == "bpo")
    P = np.asarray(t.column("protein_accession").to_pylist())[m]
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
    Y = (t.column("label").to_numpy(zero_copy_only=False).astype(np.int64)[m] > 0).astype(np.float32)
    S = (np.asarray(t.column("snapshot_pair").to_pylist())[m] if "snapshot_pair" in cols
         else np.full(len(P), "eval"))
    k = np.ones(len(P), bool)
    if part == "past":
        k = S != VAL_PAIR
    elif part == "val":
        k = S == VAL_PAIR
    return P[k], G[k], Y[k]


trP, trG, trY = rows_of(DS / "train.parquet", "past")
vaP, vaG, vaY = rows_of(DS / "train.parquet", "val")
print(f"pool rows with a label: train {len(trP):,}/{trY.sum():,.0f}pos | valid {len(vaP):,}/{vaY.sum():,.0f}pos  ({time.time()-t0:.0f}s)", flush=True)

need = sorted(set(trP.tolist()) | set(vaP.tolist()))
print(f"building the classifier frame for {len(need):,} training proteins...", flush=True)
Xf = frame(need)
fpos = {a: i for i, a in enumerate(need)}
print(f"  frame {Xf.shape}, coverage {(np.abs(Xf).sum(1) > 0).mean():.1%}  ({time.time()-t0:.0f}s)", flush=True)
Xn = torch.tensor((Xf - mu) / sd, dtype=torch.float32)
with torch.no_grad():
    LOG = np.vstack([net(Xn[i:i + 256].to(DEV)).cpu().numpy() for i in range(0, len(Xn), 256)])
del Xn, Xf
print(f"  train logits {LOG.shape}  ({time.time()-t0:.0f}s)", flush=True)


QVz = np.vstack([QV, np.zeros((1, QV.shape[1]), np.float32)])
GVz = np.vstack([GVc, np.zeros((1, GVc.shape[1]), np.float32)])


def feats(P, G, LOGm, fposm):
    """Only what exists for ANY (protein, term) pair. No kNN feature appears here by design.

    Gathered with fancy indexing rather than a per-row loop: at 800k rows a Python loop doing three
    slice assignments each is minutes of nothing.
    """
    fi = np.array([fposm.get(p, -1) for p in P])
    gi = np.array([vpos.get(g, -1) for g in G])
    qi = np.array([qpos.get(p, len(QV)) for p in P])
    ci = np.array([gpos.get(g, len(GVc)) for g in G])
    ai = np.array([apos.get(g, NA) for g in G])
    ok = (fi >= 0) & (gi >= 0)
    logit = np.zeros(len(P), np.float32)
    logit[ok] = LOGm[fi[ok], gi[ok]]
    return np.hstack([
        logit[:, None],
        QVz[qi], GVz[ci], AVc[ai],
        np.array([IA.get(g, 0.0) for g in G], np.float32)[:, None],
        np.array([len(par.get(g, ())) for g in G], np.float32)[:, None],
    ]).astype(np.float32)


# 16.5M rows x 3,275 features is 201 GiB and the first attempt tried to materialise it. Keep every
# positive and 5 negatives per positive: S exists only to RANK the extras and the submission cut is
# a quantile of its output, so the shifted base rate costs nothing that matters here. The positives
# are all kept because they are 0.80% of the rows and throwing any away is throwing away the signal.
def subsample(P, G, Y, ratio=5, seed=SEED):
    pos = np.where(Y > 0)[0]
    neg = np.where(Y == 0)[0]
    take = np.random.default_rng(seed).choice(neg, min(len(pos) * ratio, len(neg)), replace=False)
    idx = np.concatenate([pos, take])
    return P[idx], G[idx], Y[idx]


trP, trG, trY = subsample(trP, trG, trY)
vaP, vaG, vaY = subsample(vaP, vaG, vaY)
print(f"subsampled to all positives + 5x negatives: train {len(trP):,}/{trY.sum():,.0f}pos "
      f"| valid {len(vaP):,}/{vaY.sum():,.0f}pos "
      f"({len(trP)*3275*4/1e9:.1f} GB)", flush=True)
print("assembling the training features (no kNN column by construction)...", flush=True)
Xtr = feats(trP, trG, LOG, fpos)
Xva = feats(vaP, vaG, LOG, fpos)
print(f"  {Xtr.shape}  ({time.time()-t0:.0f}s)", flush=True)
_P = float(trY.sum()); _N = float(len(trY) - _P)
S_ = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
                "min_data_in_leaf": 100, "feature_fraction": 0.5, "bagging_fraction": 0.8,
                "bagging_freq": 5, "seed": SEED, "verbose": -1, "num_threads": 12, "max_bin": 63,
                "scale_pos_weight": _N / _P, "metric": "average_precision"},
               lgb.Dataset(Xtr, label=trY), num_boost_round=1500,
               valid_sets=[lgb.Dataset(Xva, label=vaY)],
               callbacks=[lgb.early_stopping(50, verbose=False)])
print(f"  S trained, best_iteration {S_.best_iteration}  ({time.time()-t0:.0f}s)", flush=True)
del LOG

# ---- the blind window ----------------------------------------------------------
GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(alt.get(t_, t_))
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
Xe = frame(eprots)
epos = {a_: i for i, a_ in enumerate(eprots)}
Xen = torch.tensor((Xe - mu) / sd, dtype=torch.float32)
with torch.no_grad():
    ELOG = np.vstack([net(Xen[i:i + 256].to(DEV)).cpu().numpy() for i in range(0, len(Xen), 256)])
del Xen, Xe
vb = [i for i, g in enumerate(vocab) if g in BP]
vbg = [vocab[i] for i in vb]
print(f"  blind logits {ELOG.shape}; BP terms in vocab {len(vb):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- SEMANTIC candidate generator: top-k BP-with-def by cosine(abstract, definition) ------------
# Read-only cached abstracts (fetch_eval_abstracts.py). A term is a candidate if ANY pre-t0 paper of
# the protein is semantically about it. Only the top-k NOT in the pool become extras.
import re as _re
CACHE = W / "abstract_cache"
_name, _def = {}, {}
_cur = None
for _line in open(OBO):
    _line = _line.rstrip("\n")
    if _line == "[Term]":
        _cur = None
    elif _line.startswith("id: GO:"):
        _cur = _line[4:]
    elif _cur and _line.startswith("name: "):
        _name[_cur] = _line[6:]
    elif _cur and _line.startswith("def: "):
        _m = _re.search(r'"([^"]+)"', _line)
        if _m:
            _def[_cur] = _m.group(1)
OBO_DEF = {g: f"{_name.get(g,'')}. {_def[g]}" for g in _def}   # "name. definition", matches phase 0.5
BP_DEF = sorted(g for g in BP if g in OBO_DEF)

prot_pmids = json.load(open(W / "eval_prot_pmids.json"))
abstracts = {}
for _xmlf in CACHE.glob("pubmed_*.xml"):
    _xml = _xmlf.read_text()
    for _art in _re.split(r"<PubmedArticle>", _xml)[1:]:
        _pm = _re.search(r"<PMID[^>]*>(\d+)</PMID>", _art)
        _ab = " ".join(_re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", _art, _re.S))
        _ti = _re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", _art, _re.S)
        if _pm:
            abstracts[_pm.group(1)] = _re.sub(r"<[^>]+>", " ", (_ti.group(1) if _ti else "") + " " + _ab).strip()
print(f"  semantic: {len(BP_DEF):,} BP defs, {len(abstracts):,} cached abstracts  ({time.time()-t0:.0f}s)", flush=True)

from sentence_transformers import SentenceTransformer
_enc = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=DEV)
DEFV = _enc.encode([f"{OBO_DEF[g]}" for g in BP_DEF], batch_size=256, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
_used = sorted({p for pr in eprots for p in prot_pmids.get(pr, []) if p in abstracts})
_pmidx = {p: i for i, p in enumerate(_used)}
PMV = (_enc.encode([abstracts[p][:2000] for p in _used], batch_size=128, normalize_embeddings=True,
                   show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
       if _used else np.zeros((0, DEFV.shape[1]), np.float32))
print(f"  semantic: encoded {len(BP_DEF):,} defs + {len(_used):,} abstracts  ({time.time()-t0:.0f}s)", flush=True)

def semantic_extras(topk):
    """{prot: set(top-k BP terms by max-cosine to the protein's abstracts, not already pooled)}."""
    out = {}
    for prot in eprots:
        pms = [p for p in prot_pmids.get(prot, []) if p in _pmidx]
        if not pms:
            continue
        sub = PMV[[_pmidx[p] for p in pms]]
        sc = (DEFV @ sub.T).max(axis=1)
        k = min(topk, len(BP_DEF) - 1)
        cand = np.argpartition(-sc, k)[:k]
        pl = pool[prot]
        out[prot] = [BP_DEF[j] for j in cand if BP_DEF[j] not in pl]
    return out

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


def cafa(rows):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows:
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, ts in GT.items():
                for g_ in ts:
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


base = [(p_, g_, float(s_)) for p_, g_, s_ in zip(eP, eG, eS) if s_ > TAU]
fa = cafa(base)
# The anchor is 0.22288, not 0.2131. TAU = 0.393 makes arm A the PREFILTERED arm, which
# `build_submission.py` measures at 0.22288 on the board's line; 0.2131 is the `guard_only` arm that
# submits everything above zero. The first run set the anchor to 0.2131, got 0.22282, and its own
# precondition voided it. The precondition was right and the constant was wrong: 0.22282 against
# 0.22288 is an independent reproduction of PKW.8 to four decimals.
ANCHOR = 0.22288
res = {"anchor_prefiltered_arm": ANCHOR, "A_pool_only": fa,
       "anchor_note": "TAU=0.393 => arm A IS the prefiltered arm (build_submission.py: 0.22288). "
                      "0.2131 is guard_only and was the wrong anchor for this arm.",
       "PRECONDITION": bool(fa is not None and abs(fa - ANCHOR) < 0.002),
       "generator": "classifier_6plm_asl: candidates true 11.59% (top5) / 9.84% (top20); the bar was "
                    "1.85% (co-occurrence) and 0.19% (DAG descent).",
       "scorer_S": "LightGBM over ONLY the features defined for any pair: the full-GO logit, the "
                   "d8979601 protein code, the GO sparse code, anc2vec, term IA, parent count. "
                   "Trained on the POOL rows of v160..v225, early-stopped on v225-v227.",
       "gate": "(B - A) must clear 0.0034 AND beat C (+0.0025, already inside the noise).",
       "arms": {}}
print(f"\n  A pool only: {fa}  (prefiltered anchor {ANCHOR}, precondition {res['PRECONDITION']})"
      f"  ({time.time()-t0:.0f}s)", flush=True)
if not res["PRECONDITION"]:
    print("  PRECONDITION FAILED: the harness does not reproduce the deployed scores. VOID.", flush=True)
    json.dump(res, open(W / "score_the_extras_deep.json", "w"), indent=1)
    raise SystemExit(0)

def classifier_extras(topk):
    """{prot: [top-k classifier BP extras not in the pool]} - the reference generator (+0.02245)."""
    out = {}
    for i, prot in enumerate(eprots):
        pl = pool[prot]
        top = np.argpartition(-ELOG[i][vb], min(topk, len(vb) - 1))[:topk]
        out[prot] = [vbg[j] for j in top if vbg[j] not in pl]
    return out

# ---- MATCHED-VOLUME complementarity test -------------------------------------------------------
# The per-arm quantile submission confounds source with volume (the union submits 3-6x more extras,
# so the fill tax buries it). Fair test: pool a GENEROUS candidate set from each source (top400),
# score EVERY candidate with the same S, tag each by source, then for a FIXED budget V submit the
# global top-V by S from (classifier-only) vs (union of both). Same V, same scorer => the only
# difference is whether S ever prefers a semantic candidate over a classifier one. If union@V beats
# classifier@V, the text channel contributes candidates the classifier misses that S ranks highly.
GEN = 300
cls = classifier_extras(GEN)
sem = semantic_extras(GEN)
# one row per (prot, term, source); a term proposed by BOTH is kept once, source="both"
cand = {}   # (prot, term) -> source
for prot in eprots:
    for g in cls.get(prot, []):
        cand[(prot, g)] = "cls"
    for g in sem.get(prot, []):
        cand[(prot, g)] = "both" if (prot, g) in cand else "sem"
keys = list(cand.keys())
ex_p = np.array([k[0] for k in keys]); ex_g = np.array([k[1] for k in keys])
src = np.array([cand[k] for k in keys])
# BATCHED feats()+predict: the full (2M+, 3275) matrix is ~30GB and thrashes swap. Score in chunks so
# peak memory is one chunk (~2.6GB at 200k rows). Same result, bounded RAM.
CH = 200_000
SC = np.empty(len(ex_p), np.float32)
for s in range(0, len(ex_p), CH):
    Xc = feats(ex_p[s:s + CH], ex_g[s:s + CH], ELOG, epos)
    SC[s:s + CH] = S_.predict(Xc, num_iteration=S_.best_iteration)
    del Xc
print(f"  scored {len(ex_p):,} candidates in {(len(ex_p)+CH-1)//CH} chunks  ({time.time()-t0:.0f}s)", flush=True)
np.savez(W / "union_candidates_scored.npz", prot=ex_p, term=ex_g, src=src, score=SC)
in_cls = (src == "cls") | (src == "both")      # candidates the classifier proposes
in_sem = (src == "sem") | (src == "both")      # candidates the semantic encoder proposes
n_cls_only = int((src == "cls").sum()); n_sem_only = int((src == "sem").sum())
n_both = int((src == "both").sum())
print(f"  candidates: classifier-only {n_cls_only:,} | semantic-only {n_sem_only:,} | "
      f"both {n_both:,} (overlap {n_both/max(1,in_cls.sum()):.1%} of classifier)  "
      f"({time.time()-t0:.0f}s)", flush=True)


def submit_topV(mask, V):
    """Submit the global top-V candidates by S among those in `mask`; run cafaeval."""
    idx = np.where(mask)[0]
    if len(idx) > V:
        idx = idx[np.argpartition(-SC[idx], V)[:V]]
    rows = base + [(ex_p[i], ex_g[i], float(SC[i])) for i in idx]
    return cafa(rows), len(idx)


res["matched_volume"] = {}
# sweep budgets around the classifier's best submitted volume (~113k at top100)
for V in (80000, 113000, 160000, 226000):
    fc, nc = submit_topV(in_cls, V)
    fu, nu = submit_topV(in_cls | in_sem, V)   # union universe, same budget
    fs, nsv = submit_topV(in_sem, V)
    row = {"classifier": fc, "classifier_delta": round(fc - fa, 5) if fc else None,
           "union": fu, "union_delta": round(fu - fa, 5) if fu else None,
           "semantic": fs, "semantic_delta": round(fs - fa, 5) if fs else None,
           "union_minus_classifier": round(fu - fc, 5) if (fu and fc) else None, "V": V}
    res["matched_volume"][f"V{V}"] = row
    print(f"  V={V:>6,}: classifier {fc} ({row['classifier_delta']:+.5f}) | "
          f"union {fu} ({row['union_delta']:+.5f}) | semantic {fs} ({row['semantic_delta']:+.5f}) | "
          f"union-cls {row['union_minus_classifier']:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "union_matched_volume.json", "w"), indent=1)

best_uc = max((r["union_minus_classifier"] for r in res["matched_volume"].values()
               if r.get("union_minus_classifier") is not None), default=None)
res["verdict"] = {"best_union_minus_classifier_matched": best_uc,
                  "SEMANTIC_ADDS_AT_MATCHED_VOLUME": bool(best_uc is not None and best_uc > 0.0034)}
json.dump(res, open(W / "union_matched_volume.json", "w"), indent=1)
print(f"\n=== does the text channel add candidates the classifier misses, at matched volume? ===", flush=True)
print(f"  best (union - classifier) at matched V: {best_uc:+.5f}", flush=True)
print(f"  semantic adds at matched volume (>0.0034): "
      f"{res['verdict']['SEMANTIC_ADDS_AT_MATCHED_VOLUME']}", flush=True)
print("DONE", flush=True)
