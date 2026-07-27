"""CROSS-MODALITY CONSENSUS -- Phase 0: the consensus precision curve.

The author's hypothesis: each orthogonal channel floods low-precision alone, but a BP term proposed by
MULTIPLE ORTHOGONAL modalities (independent errors) compounds to high precision. This differs from the
already-tested multi-PLM agreement (MULTIPLM_POOL, agreement WITHIN the sequence modality, correlated
errors). Here the three modalities are ORTHOGONAL:
  M_seq : sequence-kNN, the PLM FAMILY as ONE modality (term proposed if >=1 of 6 PLMs proposes it)
  M_net : the STRING network (net_prop_clean)
  M_clf : the full-BP-vocab classifier (classifier_6plm_asl), top-K BP-toi terms per protein
Phylo profiling (4th modality) has NO proposals yet (storage/phylo_profile/ still building) -> 3 modalities.

Classifier proposals are computed from the CACHED generator_frames (6 raw PLM embeddings, 88,212 prots)
so NO live DB is touched. The 8320-d frame is concatenated in the classifier's canonical PLM order,
z-scored by the checkpoint's mu/sd, run through the MLP, restricted to BP-toi columns, top-K per protein.

The curve: for LK-BPO and PK-BPO, over BP candidates NOT in the deployed pool (PK: also not in known),
precision (raw + IA-weighted) and true IA-mass as a function of the number of orthogonal modalities
agreeing (1,2,3). Single-channel bars for context: co-occ 1%, network 5.8%, classifier 11.6%.

Frame = full board window Sep_2025_Mar_2026 (matches the single-channel bars). GT propagated to BP closure.
READ-ONLY, frozen data, writes only under storage/consensus/.
"""
import json, pickle, collections, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
SC = ROOT / "storage/regen_headline/recall_scratch"
GF = ROOT / "storage/cooc_experiment/generator_frames"
CKPT = ROOT / "storage/fullgo_models/classifier_6plm_asl.pt"
LAB = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
REL = ROOT / "CAFA_forever/data/releases/Sep_2025_Mar_2026"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
OUT = ROOT / "storage/consensus"
CLF_TOPK = 500   # classifier proposal size (recall-heavy, comparable to seq/net medians)
DIVERSE = ["ankh_base", "ankh_large", "esm2_150m", "esm2_650m", "esm2_3b", "esmc_600m"]
# frame order = classifier canonical PLM order (score_the_extras ORDER), files in generator_frames
FRAME_ORDER = [("raw_ankh_base", 768), ("raw_esm2_3b", 2560), ("raw_ankh_large", 1536),
               ("raw_esm2_650m", 1280), ("raw_esmc_600m", 1152), ("raw_prott5", 1024)]

# ---- ontology / IA -------------------------------------------------------------
par = collections.defaultdict(set); nsd = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): nsd[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in nsd.items() if n == "biological_process"}
_AC = {}
def anc(t):
    if t in _AC: return _AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    _AC[t] = o; return o
def clo_bp(ts):
    o = set()
    for g in ts:
        g = alt.get(g, g)
        if g in BP: o.add(g)
        for a in anc(g):
            if a in BP: o.add(a)
    return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass

# ---- toi vocab (all BP) + kNN/net caches ---------------------------------------
toi_go = [alt.get(t, t) for t in json.load(open(SC / "toi_vocab.json"))["toi_go"]]
caches = {n: {str(k): v for k, v in pickle.load(open(SC / f"knn_prop_{n}.pkl", "rb")).items()} for n in DIVERSE}
net_raw = {str(k): v for k, v in pickle.load(open(SC / "net_prop_clean.pkl", "rb")).items()}
log(f"loaded {len(DIVERSE)} PLM caches + STRING ({len(net_raw)} prots); toi_go {len(toi_go)} (all BP)")

def seq_set(p):
    u = set()
    for n in DIVERSE:
        v = caches[n].get(p)
        if v is not None: u.update(int(i) for i in v.tolist())
    return {toi_go[i] for i in u}
def net_set(p):
    v = net_raw.get(p)
    return {toi_go[i] for i in v.tolist()} if v is not None else set()

# ---- classifier proposals from CACHED frames (NO DB) ---------------------------
gf_accs = json.load(open(GF / "accs.json"))
gf_pos = {a: i for i, a in enumerate(gf_accs)}
query_prots = sorted(set(caches["ankh_base"].keys()))
qidx = np.array([gf_pos[p] for p in query_prots])   # all covered (verified)
log(f"query proteins {len(query_prots)}; building 8320-d frame from cached npy for {len(qidx)} rows")
cols = []
for name, dim in FRAME_ORDER:
    arr = np.load(GF / f"{name}.npy", mmap_mode="r")
    cols.append(np.asarray(arr[qidx], dtype=np.float32))
X = np.concatenate(cols, axis=1)
assert X.shape[1] == 8320, X.shape
log(f"frame {X.shape} coverage {(np.abs(X).sum(1) > 0).mean():.1%}")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
vocab = [alt.get(g, g) for g in ck["vocab"]]
mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
net_mlp = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(1024, len(vocab)))
miss, _ = net_mlp.load_state_dict({k[4:] if k.startswith("net.") else k: v
                                   for k, v in ck["state_dict"].items()}, strict=False)
assert not miss, miss
net_mlp.eval()
# BP-toi columns of the classifier output
bp_toi = set(toi_go)
vb = [i for i, g in enumerate(vocab) if g in bp_toi]
vbg = np.array([vocab[i] for i in vb])
log(f"classifier {len(vocab)} labels -> {len(vb)} BP-toi columns; top{CLF_TOPK} per protein")
Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
clf_prop = {}
with torch.no_grad():
    for s in range(0, len(Xn), 512):
        lg = net_mlp(Xn[s:s + 512]).numpy()[:, vb]
        k = min(CLF_TOPK, lg.shape[1])
        top = np.argpartition(-lg, k - 1, axis=1)[:, :k]
        for r, prot in enumerate(query_prots[s:s + 512]):
            clf_prop[prot] = set(vbg[top[r]].tolist())
del Xn, X, cols
log(f"classifier proposals built for {len(clf_prop)} proteins")
pickle.dump({p: sorted(s) for p, s in clf_prop.items()}, open(OUT / "clf_prop_topk.pkl", "wb"))

# ---- deployed pool + known (per cell) ------------------------------------------
def load_pool(short):
    pool = collections.defaultdict(set)
    for line in open(LAB / "percut_rerank/predictions" / short / f"{short}.tsv"):
        x = line.rstrip("\n").split("\t")
        if len(x) >= 2: pool[x[0]].add(alt.get(x[1], x[1]))
    return pool
def load_gt(cell):
    gt = collections.defaultdict(set)
    with open(REL / f"groundtruth_{cell}.tsv") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                gt[f[0]] |= clo_bp([f[1]])
    return gt
def load_known(cell):
    kn = collections.defaultdict(set)
    kf = REL / f"groundtruth_{cell}_known.tsv"
    if kf.exists():
        with open(kf) as fh:
            next(fh)
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3 and f[2] == "P": kn[f[0]] |= clo_bp([f[1]])
    return kn

report = {"classifier_topk": CLF_TOPK, "modalities": ["seq(6PLM union)", "net(STRING)", "clf(classifier)"],
          "phylo": "PENDING (no proposals; storage/phylo_profile still building) -> 3 modalities",
          "single_channel_bars": {"cooccurrence": 0.01, "network": 0.058, "classifier": 0.116},
          "frame": "full board Sep_2025_Mar_2026, BP closure, PK -known", "cells": {}}

for cell in ("LK", "PK"):
    short = cell.lower()
    pool = load_pool(short); gt = load_gt(cell); known = load_known(cell)
    dep_prots = set(pool)
    # candidate rows over proteins present in deployed submission AND with a cache
    prots = sorted(dep_prots & set(caches["ankh_base"]) & set(gt))
    rows = []   # (consensus_level, in_seq, in_net, in_clf, is_true, ia)
    per_mod = {"seq": [0, 0, 0.0, 0.0], "net": [0, 0, 0.0, 0.0], "clf": [0, 0, 0.0, 0.0]}  # n,ntrue,ia_all,ia_true
    for p in prots:
        pl = pool.get(p, set()); kn = known.get(p, set()) if cell == "PK" else set()
        gtp = gt.get(p, set())
        ss = seq_set(p); ns_ = net_set(p); cs = clf_prop.get(p, set())
        cand = (ss | ns_ | cs) - pl - kn
        for term in cand:
            insq, inn, inc = term in ss, term in ns_, term in cs
            lvl = int(insq) + int(inn) + int(inc)
            ist = term in gtp
            ia = IA.get(term, 0.0)
            rows.append((lvl, insq, inn, inc, ist, ia))
            for tag, has in (("seq", insq), ("net", inn), ("clf", inc)):
                if has:
                    per_mod[tag][0] += 1; per_mod[tag][1] += int(ist)
                    per_mod[tag][2] += ia; per_mod[tag][3] += ia if ist else 0.0
    R = np.array(rows, dtype=object)
    lvl = np.array([r[0] for r in rows]); ist = np.array([r[4] for r in rows]); ia = np.array([r[5] for r in rows], float)
    curve = {}
    for L in (1, 2, 3):
        m = lvl == L
        n = int(m.sum()); nt = int(ist[m].sum())
        ia_all = float(ia[m].sum()); ia_true = float(ia[(lvl == L) & ist].sum())
        curve[f"agree{L}"] = {"n_cand": n, "n_true": nt,
                              "precision_raw": round(nt / n, 4) if n else None,
                              "precision_ia_weighted": round(ia_true / ia_all, 4) if ia_all else None,
                              "true_ia_mass": round(ia_true, 1)}
    cum = {}
    for L in (1, 2, 3):
        m = lvl >= L
        n = int(m.sum()); nt = int(ist[m].sum())
        ia_all = float(ia[m].sum()); ia_true = float(ia[(lvl >= L) & ist].sum())
        cum[f"agree_ge{L}"] = {"n_cand": n, "n_true": nt,
                               "precision_raw": round(nt / n, 4) if n else None,
                               "precision_ia_weighted": round(ia_true / ia_all, 4) if ia_all else None,
                               "true_ia_mass": round(ia_true, 1)}
    single = {tag: {"n_cand": v[0], "n_true": v[1], "precision_raw": round(v[1] / v[0], 4) if v[0] else None,
                    "precision_ia_weighted": round(v[3] / v[2], 4) if v[2] else None, "true_ia_mass": round(v[3], 1)}
              for tag, v in per_mod.items()}
    total_true_ia = float(ia[ist].sum())
    report["cells"][cell + "_BPO"] = {"n_proteins": len(prots), "n_candidates_total": len(rows),
                                      "total_true_ia_mass_available": round(total_true_ia, 1),
                                      "single_channel_precision": single,
                                      "by_consensus_level": curve, "by_consensus_ge": cum}
    log(f"{cell}: {len(prots)} prot, {len(rows):,} cand | "
        + " ".join(f"L{L}:prec={curve[f'agree{L}']['precision_raw']},iaM={curve[f'agree{L}']['true_ia_mass']}" for L in (1, 2, 3)))
    json.dump(report, open(OUT / "phase0_precision_curve.json", "w"), indent=1)

json.dump(report, open(OUT / "phase0_precision_curve.json", "w"), indent=1)
log("PHASE 0 DONE")
