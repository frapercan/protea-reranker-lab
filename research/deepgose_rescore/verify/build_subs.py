#!/usr/bin/env python
"""Fresh independent build of anchor / B_SE / control submissions.
READ-ONLY on all inputs. Writes only under verify/ dirs.
"""
import os
import numpy as np

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
LK = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions/lk/lk.tsv"
NPZ = "/home/frapercan/Thesis2/storage/deepgose_kwta/se_scores_LK.npz"
OUT = "/home/frapercan/Thesis2/storage/deepgose_rescore/verify"

# ---- Parse OBO: alt_id -> canonical, and namespace of each canonical term ----
alt2canon = {}
namespace = {}  # canonical id -> namespace
cur_id = None
cur_ns = None
cur_alts = []
def flush():
    global cur_id, cur_ns, cur_alts
    if cur_id is not None:
        if cur_ns is not None:
            namespace[cur_id] = cur_ns
        for a in cur_alts:
            alt2canon[a] = cur_id
    cur_id, cur_ns, cur_alts = None, None, []

with open(OBO) as f:
    in_term = False
    for line in f:
        line = line.rstrip("\n")
        if line == "[Term]":
            flush()
            in_term = True
            continue
        if line.startswith("[") and line.endswith("]") and line != "[Term]":
            flush()
            in_term = False
            continue
        if not in_term:
            continue
        if line.startswith("id:"):
            cur_id = line[3:].strip()
        elif line.startswith("namespace:"):
            cur_ns = line[10:].strip()
        elif line.startswith("alt_id:"):
            cur_alts.append(line[7:].strip())
flush()

def canon(t):
    return alt2canon.get(t, t)

BP_NS = "biological_process"
def is_bp(t):
    c = canon(t)
    return namespace.get(c) == BP_NS

print(f"[obo] canonical terms: {len(namespace)}  alt_ids: {len(alt2canon)}")
bp_terms_total = sum(1 for v in namespace.values() if v == BP_NS)
print(f"[obo] BP canonical terms: {bp_terms_total}")

# ---- Load SE scores, build (canon_protein? proteins are UniProt, canon_term) -> score ----
z = np.load(NPZ)
se_prots = [str(p) for p in z["proteins"]]
se_terms_raw = [str(t) for t in z["terms"]]
se_avg = z["avg"]  # (523, 19723)
prot_idx = {p: i for i, p in enumerate(se_prots)}
# term column index by canonical term id
se_term_canon = [canon(t) for t in se_terms_raw]
term_col = {}
for j, ct in enumerate(se_term_canon):
    # if duplicate canonical after alt mapping, keep first (shouldn't happen materially)
    if ct not in term_col:
        term_col[ct] = j
print(f"[se] proteins {len(se_prots)} terms {len(se_terms_raw)} unique-canon-terms {len(term_col)}")

def se_score(prot, term):
    ci = prot_idx.get(prot)
    if ci is None:
        return None
    cj = term_col.get(canon(term))
    if cj is None:
        return None
    return float(se_avg[ci, cj])

# ---- Read anchor rows verbatim ----
rows = []  # (prot, term_raw, score_str, score_float)
with open(LK) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        p, t, s = line.split("\t")
        rows.append((p, t, s, float(s)))
print(f"[anchor] rows {len(rows)}")

# BP rows (by namespace after alt mapping)
bp_flags = [is_bp(t) for (_, t, _, _) in rows]
n_bp = sum(bp_flags)
print(f"[anchor] BP rows {n_bp}  non-BP rows {len(rows)-n_bp}")

# ---- Write anchor dir (verbatim) ----
def write_dir(dirname, out_rows):
    d = os.path.join(OUT, dirname)
    os.makedirs(d, exist_ok=True)
    # clean any old tsv
    for fn in os.listdir(d):
        if fn.endswith(".tsv"):
            os.remove(os.path.join(d, fn))
    p = os.path.join(d, "lk.tsv")
    with open(p, "w") as fo:
        for (prot, term, s) in out_rows:
            fo.write(f"{prot}\t{term}\t{s}\n")
    return d

anchor_rows = [(p, t, s) for (p, t, s, _) in rows]
write_dir("anchor", anchor_rows)

# ---- B_SE: replace BP-row score with SE score (0.0 if missing) ----
se_vals_for_bp = []  # collect the SE-assigned values on BP rows (for control shuffle)
bse_rows = []
n_real = 0
n_zero = 0
for (p, t, s, sf), isbp in zip(rows, bp_flags):
    if isbp:
        v = se_score(p, t)
        if v is None:
            v = 0.0
            n_zero += 1
        else:
            n_real += 1
        se_vals_for_bp.append(v)
        bse_rows.append((p, t, repr(v)))
    else:
        bse_rows.append((p, t, s))
assert len(bse_rows) == len(rows), "row count mismatch B_SE"
# row set identity check
anchor_pairs = set((p, t) for (p, t, _, _) in rows)
bse_pairs = set((p, t) for (p, t, _) in bse_rows)
assert anchor_pairs == bse_pairs, "pair set mismatch B_SE"
write_dir("bse", bse_rows)
print(f"[bse] BP rows with REAL se score {n_real}  set-to-0 {n_zero}")

# ---- CONTROL_random: shuffle the SE values across BP rows ----
rng = np.random.default_rng(12345)
perm = rng.permutation(len(se_vals_for_bp))
shuffled = [se_vals_for_bp[i] for i in perm]
ctrl_rows = []
k = 0
for (p, t, s, sf), isbp in zip(rows, bp_flags):
    if isbp:
        ctrl_rows.append((p, t, repr(shuffled[k])))
        k += 1
    else:
        ctrl_rows.append((p, t, s))
assert k == n_bp
assert len(ctrl_rows) == len(rows)
# same multiset check
assert sorted(se_vals_for_bp) == sorted(shuffled)
write_dir("control", ctrl_rows)
print(f"[control] shuffled {k} BP-row SE values")

print("BUILD_OK", len(rows), n_bp, n_real, n_zero)
