"""Which columns actually differ between the two half-rows of a duplicated candidate?

The pool emits 40,192 (protein,term) pairs TWICE: once knn-only, once classifier-only.
Before merging them we must know which features are SOURCE-SPECIFIC (must be taken from
the row that produced them) and which are PROTEIN-LEVEL (must already agree, and if they
do not, that is a second defect).

This is diagnosis, not a fix. It defines the merge rather than assuming it.
"""
import collections, json
import numpy as np, pyarrow.parquet as pq

DS = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230/eval.parquet"
EX = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair", "qualifier",
      "evidence_code", "taxonomic_relation", "aspect"}
pf = pq.ParquetFile(DS)
FEATS = [c for c in pf.schema_arrow.names if c not in EX]

t = pq.read_table(DS, columns=FEATS + ["category", "aspect", "protein_accession", "go_term_id", "label"])
c = np.asarray(t.column("category").to_pylist())
a = np.asarray(t.column("aspect").to_pylist())
m = (c == "pk") & (a == "bpo")
P = np.asarray(t.column("protein_accession").to_pylist())[m]
G = np.asarray(t.column("go_term_id").to_pylist())[m]
X = np.empty((int(m.sum()), len(FEATS)), dtype=np.float64)
for j, col in enumerate(FEATS):
    X[:, j] = t.column(col).to_numpy(zero_copy_only=False).astype(np.float64)[m]

keys = list(zip(P, G))
first = {}
pairs = []                      # (i_knn_row, i_clf_row)
kpi = FEATS.index("knn_present")
for i, k in enumerate(keys):
    if k in first:
        pairs.append((first[k], i))
    else:
        first[k] = i
print(f"duplicated pairs: {len(pairs):,}", flush=True)

ia = np.array([p[0] for p in pairs])
ib = np.array([p[1] for p in pairs])
# orient: A = the row with knn_present, B = the other
knn_first = X[ia, kpi] > 0
lo = np.where(knn_first, ia, ib)      # the KNN half
hi = np.where(knn_first, ib, ia)      # the CLASSIFIER half

rows = []
for j, col in enumerate(FEATS):
    va, vb = X[lo, j], X[hi, j]
    both_nan = np.isnan(va) & np.isnan(vb)
    diff = ~both_nan & ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
    if diff.sum() == 0:
        continue
    rows.append({
        "feature": col,
        "differs_pct": round(float(diff.mean() * 100), 1),
        "knn_half_nan_pct": round(float(np.isnan(va).mean() * 100), 1),
        "clf_half_nan_pct": round(float(np.isnan(vb).mean() * 100), 1),
        "knn_half_zero_pct": round(float((va == 0).mean() * 100), 1),
        "clf_half_zero_pct": round(float((vb == 0).mean() * 100), 1),
    })
rows.sort(key=lambda r: -r["differs_pct"])
print(f"\n{'feature':32s} {'differs%':>9} {'knnNaN%':>8} {'clfNaN%':>8} {'knn0%':>7} {'clf0%':>7}")
for r in rows:
    print(f"{r['feature']:32s} {r['differs_pct']:>9} {r['knn_half_nan_pct']:>8} "
          f"{r['clf_half_nan_pct']:>8} {r['knn_half_zero_pct']:>7} {r['clf_half_zero_pct']:>7}")
same = [c for c in FEATS if c not in {r["feature"] for r in rows}]
print(f"\nIDENTICAL in both halves ({len(same)} features, protein-level as expected):")
print("  " + ", ".join(same))
json.dump({"n_pairs": len(pairs), "differing": rows, "identical": same},
          open("/home/frapercan/Thesis2/storage/cooc_experiment/merge_diag.json", "w"), indent=1)
print("\nDONE", flush=True)
