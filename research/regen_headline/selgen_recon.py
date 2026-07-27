"""SELECTIVE GENERATION SUBMISSION -- Stage 0 reconnaissance (NO cafaeval).

Assemble the classifier's BP extra candidates (DB-FREE from generator_frames), label them against
the board propagated ground truth, and measure PER-STRATUM precision. The bet: precision is
concentrated, so excluding low-precision strata and submitting only high-precision strata could net
positive under prop=fill. Before spending cafaeval on a threshold sweep, check whether ANY stratum
has enough precision AND volume to plausibly clear the fill-tax break-even.

Everything here is t0-derived (leakage-safe): the classifier is v227-trained, its input frame is the
cached v227 embeddings, IA / term-freq / depth are t0, known_count is t0 knowns, taxon is static.
The only post-t0 information is the candidate LABEL (the board GT = the target), exactly as intended.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/regen_headline")
GF = Path("/home/frapercan/Thesis2/storage/cooc_experiment/generator_frames")
LAB = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
GT_DIR = LAB / "lafa_gt"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
EVAL_PARQUET = LAB / "percut_rerank" / "eval.parquet"
TAXON_F = W / "eval_acc_taxon.tsv"
K_CAND = 50  # candidate pool depth per protein (top-K BP logits not in pool)

# ---- ontology -----------------------------------------------------------------
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
_AC = {}
def anc(t):
    if t in _AC:
        return _AC[t]
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        for p in par.get(x, ()):
            if p not in seen:
                seen.add(p); stack.append(p)
    _AC[t] = seen
    return seen
# depth = longest path to a root (no BP parent)
_DEPTH = {}
def depth(t):
    if t in _DEPTH:
        return _DEPTH[t]
    ps = [p for p in par.get(t, ()) if p in BP]
    d = 0 if not ps else 1 + max(depth(p) for p in ps)
    _DEPTH[t] = d
    return d

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass

# ---- classifier + DB-free frame ----------------------------------------------
ck = torch.load("/home/frapercan/Thesis2/storage/fullgo_models/classifier_6plm_asl.pt",
                map_location="cpu", weights_only=False)
vocab = [alt.get(g, g) for g in ck["vocab"]]
mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
net = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, len(vocab)))
miss, _ = net.load_state_dict({k[4:] if k.startswith("net.") else k: v
                               for k, v in ck["state_dict"].items()}, strict=False)
assert not miss, miss
net.eval()
vpos = {g: i for i, g in enumerate(vocab)}
vb = [i for i, g in enumerate(vocab) if g in BP]
vbg = [vocab[i] for i in vb]
print(f"classifier {len(vocab):,} labels, BP {len(vb):,}  ({time.time()-t0:.0f}s)", flush=True)

accs = json.load(open(GF / "accs.json"))
apos = {a: i for i, a in enumerate(accs)}
ORDER = ["raw_ankh_base", "raw_esm2_3b", "raw_ankh_large", "raw_esm2_650m", "raw_esmc_600m", "raw_prott5"]
X = np.hstack([np.load(GF / f"{n}.npy") for n in ORDER]).astype(np.float32)
assert X.shape[1] == 8320, X.shape
Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
del X
with torch.no_grad():
    LOGbp = np.vstack([net(Xn[i:i + 512]).cpu().numpy()[:, vb] for i in range(0, len(Xn), 512)])
del Xn
print(f"BP logits {LOGbp.shape}  ({time.time()-t0:.0f}s)", flush=True)

# ---- per-protein t0 features: known_count, taxon; per-term t0: freq ------------
te = pq.read_table(EVAL_PARQUET, columns=["protein_accession", "category", "aspect",
                                          "anc2vec_query_known_count", "go_term_id", "go_term_frequency"])
ec = np.asarray(te.column("category").to_pylist()); ea = np.asarray(te.column("aspect").to_pylist())
ep = np.asarray(te.column("protein_accession").to_pylist())
ek = te.column("anc2vec_query_known_count").to_numpy(zero_copy_only=False)
eg = np.asarray(te.column("go_term_id").to_pylist())
ef = te.column("go_term_frequency").to_numpy(zero_copy_only=False)
known_count = {}
cell_prots = {"lk": set(), "pk": set()}
for p, c, a, k in zip(ep, ec, ea, ek):
    if a == "bpo" and c in ("lk", "pk"):
        known_count[p] = float(k) if k == k else 0.0
        cell_prots[c].add(p)
TF = {}
for g, f in zip(eg, ef):
    g2 = alt.get(g, g)
    if f == f:
        TF[g2] = float(f)
print(f"cell prots lk {len(cell_prots['lk']):,} pk {len(cell_prots['pk']):,}  ({time.time()-t0:.0f}s)", flush=True)

taxon = {}
if TAXON_F.exists():
    for line in open(TAXON_F):
        if line.startswith("DONE"):
            continue
        pp = line.rstrip("\n").split("\t")
        if len(pp) == 2:
            taxon[pp[0]] = pp[1]
taxon_density = collections.Counter(taxon.values())  # eval proteins per taxon = density proxy
print(f"taxon map {len(taxon):,} proteins, {len(taxon_density):,} taxa  ({time.time()-t0:.0f}s)", flush=True)

# ---- board GT (propagated) ----------------------------------------------------
def load_gt(fn):
    gt = collections.defaultdict(set)
    with open(fn) as fh:
        head = fh.readline()
        # files may or may not have header; detect
        first = head.rstrip("\n").split("\t")
        def add(pp, tt):
            t2 = alt.get(tt, tt)
            gt[pp].add(t2); gt[pp].update(anc(t2))
        if first and first[0].startswith(("GO:",)) is False and len(first) >= 2 and first[1].startswith("GO:"):
            add(first[0], first[1])
        for line in fh:
            x = line.rstrip("\n").split("\t")
            if len(x) >= 2 and x[1].startswith("GO:"):
                add(x[0], x[1])
    return gt
GT = {"lk": load_gt(GT_DIR / "groundtruth_LK.tsv"), "pk": load_gt(GT_DIR / "groundtruth_PK.tsv")}
# PK true frame excludes already-known pairs
PK_KNOWN = collections.defaultdict(set)
with open(GT_DIR / "groundtruth_PK_known.tsv") as fh:
    for line in fh:
        x = line.rstrip("\n").split("\t")
        if len(x) >= 2 and x[1].startswith("GO:"):
            PK_KNOWN[x[0]].add(alt.get(x[1], x[1]))
print(f"GT lk {len(GT['lk']):,} pk {len(GT['pk']):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- deployed pool per cell (what is already submitted) ------------------------
def load_pool(fn):
    pool = collections.defaultdict(set)
    for line in open(fn):
        x = line.rstrip("\n").split("\t")
        if len(x) >= 3:
            pool[x[0]].add(alt.get(x[1], x[1]))
    return pool
POOL = {"lk": load_pool(LAB / "percut_rerank/predictions/lk/lk.tsv"),
        "pk": load_pool(LAB / "percut_rerank/predictions/pk/pk.tsv")}

# ---- build extras + label + stratify ------------------------------------------
def build(cell):
    prots = sorted(cell_prots[cell] & set(GT[cell]) & set(apos))
    rows = []
    for p in prots:
        i = apos[p]
        lg = LOGbp[i]
        top = np.argpartition(-lg, K_CAND)[:K_CAND]
        top = top[np.argsort(-lg[top])]
        pool = POOL[cell].get(p, set())
        gtp = GT[cell].get(p, set())
        known = PK_KNOWN.get(p, set()) if cell == "pk" else set()
        for rank, j in enumerate(top):
            g = vbg[j]
            if g in pool:
                continue
            if cell == "pk" and g in known:
                continue  # -known: already-known terms are excluded from PK scoring
            rows.append((p, g, float(lg[j]), rank,
                         1 if g in gtp else 0,
                         IA.get(g, 0.0), TF.get(g, 0.0), depth(g),
                         known_count.get(p, 0.0),
                         taxon_density.get(taxon.get(p, "?"), 0)))
    dt = np.dtype([("p", "U15"), ("g", "U12"), ("logit", "f4"), ("rank", "i4"), ("y", "i4"),
                   ("ia", "f4"), ("tf", "f4"), ("depth", "i4"), ("known", "f4"), ("taxden", "i4")])
    return np.array(rows, dtype=dt)

report = {"K_CAND": K_CAND, "deployed_ref_trueframe": {"LK_BPO": 0.31096, "PK_BPO": 0.14024},
          "cells": {}}
for cell in ("lk", "pk"):
    R = build(cell)
    n = len(R); ntrue = int(R["y"].sum())
    d = {"n_candidates": n, "n_true": ntrue, "aggregate_precision": round(ntrue / max(n, 1), 4),
         "n_proteins": len(set(R["p"].tolist()))}
    # precision by rank band (source score proxy)
    for lo, hi, tag in [(0, 5, "top5"), (0, 10, "top10"), (0, 20, "top20"), (0, 50, "top50")]:
        m = (R["rank"] >= lo) & (R["rank"] < hi)
        d[f"prec_{tag}"] = round(R["y"][m].mean(), 4) if m.sum() else None
    # precision by logit decile
    order = np.argsort(-R["logit"])
    dec = {}
    for q in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        k = max(1, int(n * q))
        idx = order[:k]
        dec[f"top_{q}"] = {"n": int(k), "prec": round(float(R["y"][idx].mean()), 4),
                           "ia_weighted_prec": round(float((R["ia"][idx] * R["y"][idx]).sum() /
                                                           max(R["ia"][idx].sum(), 1e-9)), 4)}
    d["precision_by_logit_fraction"] = dec
    # precision by IA band
    iaband = {}
    for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 99)]:
        m = (R["ia"] >= lo) & (R["ia"] < hi)
        iaband[f"IA[{lo},{hi})"] = {"n": int(m.sum()),
                                    "prec": round(float(R["y"][m].mean()), 4) if m.sum() else None}
    d["precision_by_IA_band"] = iaband
    # precision by known-count band
    kb = {}
    for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 1e9)]:
        m = (R["known"] >= lo) & (R["known"] < hi)
        kb[f"known[{lo},{hi})"] = {"n": int(m.sum()),
                                   "prec": round(float(R["y"][m].mean()), 4) if m.sum() else None}
    d["precision_by_known_count"] = kb
    # precision by taxon-density band
    tb = {}
    for lo, hi in [(0, 1), (1, 50), (50, 500), (500, 1e9)]:
        m = (R["taxden"] >= lo) & (R["taxden"] < hi)
        tb[f"taxden[{lo},{hi})"] = {"n": int(m.sum()),
                                    "prec": round(float(R["y"][m].mean()), 4) if m.sum() else None}
    d["precision_by_taxon_density"] = tb
    # top taxa by precision (>=200 candidates)
    tt = {}
    for tx in set(taxon.values()):
        m = np.array([taxon.get(p) == tx for p in R["p"]])
        if m.sum() >= 200:
            tt[tx] = {"n": int(m.sum()), "prec": round(float(R["y"][m].mean()), 4)}
    d["taxa_prec_ge200cand"] = dict(sorted(tt.items(), key=lambda kv: -kv[1]["prec"])[:12])
    report["cells"][cell.upper() + "_BPO"] = d
    np.save(W / f"selgen_cand_{cell}.npy", R)
    print(f"{cell.upper()}: {n:,} cand, {ntrue:,} true, agg prec {ntrue/max(n,1):.4f}, "
          f"top5 {d['prec_top5']}, top-1%logit {dec['top_0.01']['prec']}  ({time.time()-t0:.0f}s)", flush=True)

json.dump(report, open(W / "selgen_recon.json", "w"), indent=1)
print("RECON DONE", flush=True)
