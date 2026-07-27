"""L2 de-risk: averaged soft Pmin/Pmax propagation (ProtBoost 4.5) on the PK
validation predictions. Reuses the existing `both` PK booster (no retrain):
predict on the validation eval split, then compare cafaeval PK for:
  (A) baseline  = raw scores + prop=fill   (MUST reproduce ~0.3720 = sanity gate)
  (B) L2        = soft Pmin/Pmax per protein + prop=fill
Pmin(N)=min(Pmin(parents))*0.7+P(N)*0.3 (root->leaf); Pmax(N)=max(Pmax(children))*0.7
+P(N)*0.3 (leaf->root); P_post=(Pmin+Pmax)/2. Empty parents/children -> keep own P
(blend only applies with neighbours). Only the relative B-vs-A delta is meaningful."""
import gc
import io
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from cafaeval.evaluation import cafa_eval
from minio import Minio

BASE = "datasets/fullgo-union-SELECT-160-220-227-v5"
BUCKET = "protea"
D = "/home/frapercan/Thesis2/storage/fullgo_models/selfprior_ia_experiment"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
BOOSTER = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_both_serve/ensemble_gbm_PK.txt"
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
LEAN = [
    "distance", "identity_nw", "similarity_nw", "alignment_score_nw", "gaps_pct_nw",
    "alignment_length_nw", "identity_sw", "similarity_sw", "alignment_score_sw",
    "gaps_pct_sw", "alignment_length_sw", "length_query", "length_ref",
    "taxonomic_distance", "taxonomic_common_ancestors", "vote_count", "k_position",
    "go_term_frequency", "ref_annotation_density", "neighbor_distance_std",
    "neighbor_vote_fraction", "neighbor_min_distance", "neighbor_mean_distance",
    "knn_present", "classifier_score", "classifier_present", "self_prior_score",
    "association_total", "association_cross", "association_present",
]
SP_IDX = LEAN.index("self_prior_score")
NAMES = LEAN + ["IA"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parents_map():
    par = defaultdict(set)
    cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif line.startswith("is_a:") and cur:
            par[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            p = line.split()
            if len(p) >= 3:
                par[cur].add(p[2])
    ch = defaultdict(set)
    for c, ps in par.items():
        for p in ps:
            ch[p].add(c)
    return par, ch


def ancestors(t, par, cache):
    if t in cache:
        return cache[t]
    out = set()
    st = list(par.get(t, ()))
    while st:
        a = st.pop()
        if a in out:
            continue
        out.add(a)
        st.extend(par.get(a, ()))
    cache[t] = out
    return out


def load_eval_pk():
    raw = CLIENT.get_object(BUCKET, f"{BASE}/eval.parquet").read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    ov = np.load(f"{D}/eval_overlay.npz")
    spf_all, ia_all = ov["self_prior_fixed"], ov["IA"]
    cols = LEAN + ["category", "label", "protein_accession", "go_term_id"]
    Xs, ys, prots, gids = [], [], [], []
    off = 0
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        cat = np.asarray(b.column("category").to_numpy(zero_copy_only=False), dtype=object)
        bn = len(cat)
        m = cat == "pk"
        if m.any():
            idx = np.nonzero(m)[0]
            X = np.empty((idx.size, len(LEAN)), dtype=np.float32)
            for j, f in enumerate(LEAN):
                X[:, j] = b.column(f).to_numpy(zero_copy_only=False)[idx].astype(np.float32)
            X[:, SP_IDX] = spf_all[off:off + bn][idx]
            Xs.append(np.hstack([X, ia_all[off:off + bn][idx].reshape(-1, 1).astype(np.float32)]))
            ys.append(b.column("label").to_numpy(zero_copy_only=False)[idx].astype(np.int8))
            prots.append(np.asarray(b.column("protein_accession").to_pylist(), dtype=object)[idx])
            gids.append(np.asarray(b.column("go_term_id").to_pylist(), dtype=object)[idx])
        off += bn
    del raw, pf, ov
    gc.collect()
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(prots), np.concatenate(gids)


def soft_prop_protein(scores, par, ch, anc_cache):
    """scores: {go_id: P} for one protein's candidates. Returns {go_id: P_post}."""
    terms = set(scores)
    allt = set(terms)
    for t in terms:
        allt |= ancestors(t, par, anc_cache)  # need ancestor scores (0 if not candidate) for Pmin
    # topo order: parents before children (by ancestor-count)
    order = sorted(allt, key=lambda t: len(ancestors(t, par, anc_cache)))
    Pmin = {}
    for t in order:  # root -> leaf
        ps = [Pmin[p] for p in par.get(t, ()) if p in Pmin]
        base = scores.get(t, 0.0)
        Pmin[t] = (min(ps) * 0.7 + base * 0.3) if ps else base
    Pmax = {}
    for t in reversed(order):  # leaf -> root
        cs = [Pmax[c] for c in ch.get(t, ()) if c in Pmax]
        base = scores.get(t, 0.0)
        Pmax[t] = (max(cs) * 0.7 + base * 0.3) if cs else base
    return {t: (Pmin[t] + Pmax[t]) / 2 for t in terms}


def cafaeval_pk(pred_dict, gt):
    with tempfile.TemporaryDirectory() as td:
        pd_ = Path(td) / "pred"
        pd_.mkdir()
        with open(pd_ / "m.tsv", "w") as w:
            for (p, g), s in pred_dict.items():
                if s > 0:
                    w.write(f"{p}\t{g}\t{s:.6f}\n")
        gtf = Path(td) / "gt.tsv"
        with open(gtf, "w") as w:
            for (p, g) in sorted(gt):
                w.write(f"{p}\t{g}\n")
        df, _ = cafa_eval(OBO, str(pd_), str(gtf), ia=IA_TSV, prop="fill", norm="cafa",
                          no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
        sub = df.reset_index()
        col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
        return float(sub.groupby("ns")[col].max().mean())


log("load OBO")
par, ch = parents_map()
anc_cache = {}
log("load eval PK + booster predict")
X, y, prot, gid = load_eval_pk()
b = lgb.Booster(model_file=BOOSTER)
pred = b.predict(X, num_threads=8)

# MAX-collapse per (protein, go_id); GT = label==1
base = {}
gt = set()
by_prot = defaultdict(dict)
for p, g, s, yy in zip(prot.tolist(), gid.tolist(), pred.tolist(), y.tolist()):
    k = (p, g)
    if k not in base or s > base[k]:
        base[k] = s
    if s > by_prot[p].get(g, -1):
        by_prot[p][g] = s
    if yy == 1:
        gt.add(k)

pk_base = cafaeval_pk(base, gt)
log(f"SANITY baseline PK (raw+fill) = {pk_base:.4f}  (expect ~0.3720)")

# L2 per protein
l2 = {}
for p, scores in by_prot.items():
    post = soft_prop_protein(scores, par, ch, anc_cache)
    for g, s in post.items():
        l2[(p, g)] = s
pk_l2 = cafaeval_pk(l2, gt)
log(f"L2 soft-prop PK = {pk_l2:.4f}  vs baseline {pk_base:.4f}  delta {pk_l2-pk_base:+.4f}")
log("DONE l2 de-risk")
