"""Seed-average the M2 classifier predictions (consensus union).

Each seed run of train_classifier_m2.py emits its top-100 terms per protein. We take the
UNION of (protein, term) pairs across seeds and score each as sum(scores)/n_seeds (a seed
that did not surface the pair contributes 0). Averaging the initialisation noise of the ASL
heads lifts the classifier well beyond any single seed and generalises across frames
(SELECT 0.305 -> 0.351, TEST 0.343 -> 0.369).

Usage: seed_average.py OUT seed_pred_1.tsv seed_pred_2.tsv ...
"""
import sys
from collections import defaultdict

OUT = sys.argv[1]
PREDS = sys.argv[2:]
n = len(PREDS)
assert n >= 2, "need >=2 seed prediction files"

acc = defaultdict(float)
for path in PREDS:
    with open(path) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 3 and c[1].startswith("GO:"):
                acc[(c[0], c[1])] += float(c[2])

with open(OUT, "w") as w:
    for (p, t), s in acc.items():
        w.write(f"{p}\t{t}\t{s / n:.6f}\n")
print(f"averaged {n} seeds -> {OUT} ({len(acc)} pairs)", flush=True)
