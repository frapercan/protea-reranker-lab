"""8-PLM kNN retrieval producer for the BP recall-ceiling diagnostic (READ-ONLY, frozen).

For each PLM whose t0 (c905dffa = v227) reference embeddings are materialized in ref_cache, kNN each
LK/PK-BPO query over the t0 reference set (cosine), drop self-accession, and transfer the neighbours'
t0 BP annotations (propagated to closure, restricted to toi) as proposed candidate terms.

Leakage: reference embeddings AND reference annotations are the c905dffa=v227 snapshot (t0 =
2025-09-04 cutoff). Query vectors are looked up FROM the same t0 reference matrix by accession (the
query proteins exist in v227 with their t0 state). Self-accession neighbours are dropped so a query
cannot transfer its own annotations. No post-t0 term can enter: every transferred term is a v227
reference annotation. anno map is built once from the t0 CSR (id->GO via frozen go_term_metadata).

Output: per PLM, storage/regen_headline/recall_scratch/knn_prop_<plm>.pkl = {accession: np.int32
array of proposed toi indices}. PLM-independent (BP toi universe) so the union script can combine.
"""
import json, collections, time, pickle
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
W = ROOT / "storage/regen_headline"
SC = W / "recall_scratch"; SC.mkdir(exist_ok=True)
REFC = ROOT / "worktrees/protea-deploy/data/ref_cache"
FROZEN = ROOT / "storage/protea-frozen-v227-2025-09-04"
DS = ROOT / "repositories/protea-reranker-lab/datasets/protst-global-train227-test230/eval.parquet"
T0D = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025"
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
TOI_FILE = str(ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt/groundtruth_terms_of_interest.txt")
SNAP = "c905dffa-a5ce-430b-b17b-503e88666adb"
K = 50  # neighbours per query
CHUNK = 40000

# PLMs with a c905dffa (t0) ref_cache: 6 raw canonical + learned champion + learned chunk-attn.
PLMS = {
    "ankh_base":  "08234f06-ba76-4d7d-aaec-ae601096b4fa",
    "ankh_large": "238f79b1-3068-4c6f-9013-5cc52b4f662b",
    "esm2_150m":  "500a0c59-be09-424d-9d51-b7997629c95a",
    "esm2_650m":  "c2e9dda3-e505-4170-b50d-435a451761ac",
    "esm2_3b":    "55e43f1c-1a3b-4b1d-88c0-26b433f5f673",
    "esmc_600m":  "2bf1e753-022f-44b8-a131-9a90acb4024e",
    "learned_champion": "d8979601-ea59-4de1-9c16-21036ed67c36",
}

def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------- ontology: parents, BP, closure, toi ----------
par = collections.defaultdict(set); ns = {}; alt = {}
cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    if t in _anc: return _anc[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    _anc[t] = o; return o

# toi (BP only), index
toi_go = []
with open(TOI_FILE) as fh:
    for line in fh:
        g = line.strip()
        if g in BP: toi_go.append(g)
toi_go = sorted(set(toi_go))
toi_idx = {g: i for i, g in enumerate(toi_go)}
log(f"BP toi terms: {len(toi_go)}")

# term -> propagated toi-index array (memoized over distinct leaf terms)
_prop = {}
def prop_toi(term):
    if term in _prop: return _prop[term]
    s = set()
    j = toi_idx.get(term)
    if j is not None: s.add(j)
    for a in anc(term):
        j = toi_idx.get(a)
        if j is not None: s.add(j)
    arr = np.fromiter(s, dtype=np.int32, count=len(s)); _prop[term] = arr; return arr

# ---------- accession -> propagated BP toi-set, from t0 CSR (ankh_base c905dffa) ----------
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {int(i): g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
pre = f"{PLMS['ankh_base']}__{SNAP}"
acc0 = np.load(REFC / f"{pre}_accessions.npy")
gt0 = np.load(REFC / f"{pre}__P_anno_gtids.npy")
off0 = np.load(REFC / f"{pre}__P_anno_offsets.npy")
pidx0 = np.load(REFC / f"{pre}__P_indices.npy", allow_pickle=True).astype(np.int64)
log(f"t0 CSR: {len(pidx0)} P-annotated ref proteins, {len(gt0)} anno rows")
acc2toi = {}
for j in range(len(pidx0)):
    terms = gt0[off0[j]:off0[j + 1]]
    s = set()
    for gid in terms:
        g = id2go.get(int(gid))
        if g is None: continue
        gg = alt.get(g, g)
        if gg in BP:
            for ti in prop_toi(gg): s.add(int(ti))
    if s:
        a = acc0[pidx0[j]]
        acc2toi[a] = np.fromiter(s, dtype=np.int32, count=len(s))
log(f"acc2toi built for {len(acc2toi)} ref proteins")

# ---------- query accessions per cell (EXACT atlas universe = 523 LK / 4397 PK) ----------
qaccs = {}
for cell in ("lk", "pk"):
    recs = json.load(open(W / f"bp_atlas_perprotein_{cell}.json"))
    qaccs[cell] = sorted(set(r["protein"] for r in recs))
all_q = sorted(set(qaccs["lk"]) | set(qaccs["pk"]))
json.dump({"lk": qaccs["lk"], "pk": qaccs["pk"]}, open(SC / "query_accs.json", "w"))
log(f"queries: lk {len(qaccs['lk'])}, pk {len(qaccs['pk'])}, total {len(all_q)}")

# ---------- per-PLM kNN ----------
def run_plm(name, cfg):
    fn_out = SC / f"knn_prop_{name}.pkl"
    if fn_out.exists():
        log(f"{name}: cached, skip"); return
    pre = f"{cfg}__{SNAP}"
    racc = np.load(REFC / f"{pre}_accessions.npy")
    remb = np.load(REFC / f"{pre}_embeddings.npy", mmap_mode="r")
    N, d = remb.shape
    row_of = {a: i for i, a in enumerate(racc)}
    qrows = np.array([row_of[a] for a in all_q if a in row_of], dtype=np.int64)
    qA = np.array([a for a in all_q if a in row_of])
    Q = np.asarray(remb[qrows], dtype=np.float32)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    nq = len(qrows)
    KK = K + 5
    topv = np.full((nq, KK), -2.0, np.float32)
    topi = np.full((nq, KK), -1, np.int64)
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        R = np.asarray(remb[s:e], dtype=np.float32)
        R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
        sims = Q @ R.T  # (nq, e-s)
        gidx = np.arange(s, e)
        cv = np.concatenate([topv, sims], axis=1)
        ci = np.concatenate([topi, np.broadcast_to(gidx, (nq, e - s))], axis=1)
        part = np.argpartition(-cv, KK, axis=1)[:, :KK]
        topv = np.take_along_axis(cv, part, axis=1)
        topi = np.take_along_axis(ci, part, axis=1)
        del R, sims, cv, ci
    log(f"{name}: kNN done N={N} d={d}")
    # transfer: per query, union neighbour toi-sets (drop self accession)
    out = {}
    for qi in range(nq):
        qa = qA[qi]
        order = np.argsort(-topv[qi])
        nbrs = topi[qi][order]
        acc_set = set()
        cnt = 0
        for ri in nbrs:
            if ri < 0: continue
            na = racc[ri]
            if na == qa: continue
            t = acc2toi.get(na)
            if t is not None: acc_set.update(t.tolist())
            cnt += 1
            if cnt >= K: break
        if acc_set:
            out[qa] = np.fromiter(acc_set, dtype=np.int32, count=len(acc_set))
    pickle.dump(out, open(fn_out, "wb"))
    log(f"{name}: wrote {len(out)} query proposal sets -> {fn_out.name}")
    del Q, remb

for name, cfg in PLMS.items():
    try:
        run_plm(name, cfg)
    except Exception as ex:
        log(f"{name}: ERROR {ex!r}")
# save toi vocab for the union script
json.dump({"toi_go": toi_go}, open(SC / "toi_vocab.json", "w"))
log("DONE producer")
