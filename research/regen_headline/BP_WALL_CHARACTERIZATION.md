# The BP wall is a SIGNAL limit, not an EVIDENCE limit (measured, 2026-07-16)

Thesis-grade characterization of why PROTEA is #1 in 7/9 and not 9/9. The two
missing cells are LK-BPO and PK-BPO. This receipt answers *why*, with numbers
rather than the earlier qualitative "structural gate" argument.

## The standing (real leaderboard, same frame v227->v230, f_micro_w)

Mirror: `CAFA_forever/data/releases/Sep_2025_Mar_2026/results_{NK,LK,PK}/evaluation_best_f_micro_w.tsv`

| cell | #1 | #2 | PROTEA | gap to #1 |
|------|----|----|--------|-----------|
| NK-BPO | **PROTEA 0.3374** | percut-graft 0.3310 | (we are #1) | - |
| LK-BPO | TransFew 0.5120 | FunBind 0.4720 | **#3, 0.4402** | **+0.072** |
| PK-BPO | TransFew 0.2943 | FunBind 0.2348 | **#3, 0.2181** | **+0.076** |

We are #1 on NK-BPO. On the other two BP cells we are third, behind TransFew and
FunBind. The gap is 0.072 / 0.076, not a rounding error.

## The pool is thin, but the pool is NOT the binding constraint

PK-BPO candidate pool vs the full ground truth (41,727 true protein-BP pairs):

- true pairs reachable as candidates: **13,443 -> recall 0.322**
- per-protein recall: mean 0.327, median 0.222
- PK proteins with **zero** true BP terms in the pool: **1,492 of 4,402 (33.9%)**

Two thirds of the true terms are not candidates. The obvious reading is
"recall-limited". **That reading is wrong**, and the oracle test below is what
disproves it:

```
RERANKER (what we deliver)                 f_micro_w = 0.1255
ORACLE  (perfect ranking of the SAME pool) f_micro_w = 0.6077
  -> we capture 20.7% of what the pool we already hold allows
```

The 32%-recall pool is worth **0.6077** to a perfect ranker. We extract **0.1255**.
So the binding constraint today is **ranking, not retrieval**. This is measured on
the same harness/gt as the baseline, with the true label as the score.

It also explains, exactly, why every candidate-side lever is inert: co-occurrence
expansion (+50% relative recall) bought **+0.0021**, because more candidates do not
help a ranker that cannot order the ones it has. Same for the precision-side levers
that do not add ranking power: protst +0.0016, InterPro graft *negative* on BP
(0.140 vs 0.218).

Why the ranker cannot order them: on PK-BPO no feature we carry exceeds
**AUC 0.68** (classifier_present; then protst_text 0.64 but on 41% coverage,
classifier_score 0.63, alignment ~0.60, association_cross 0.60). Nothing separates
the 2.47% of true candidates from the rest.

## But the terms ARE predictable: 97% are in the pre-t0 vocabulary

- true PK-BP pairs whose term exists in the pre-t0 BP vocabulary: **0.970**

Only 3% are genuinely novel terms. The wall is therefore **not** an evidence
ceiling. The information exists; we fail to surface it.

## Co-occurrence recovers the recall, and it does not pay

Expanding each PK protein's candidate set from its t0-known terms via term
co-occurrence (leakage-safe: known terms are t0, co-occurrence learned on the t0
corpus, future terms absent from both):

| expansion | recall | added candidates | precision of the addition | co-occ score AUC |
|-----------|--------|------------------|---------------------------|------------------|
| pool only | 0.322 | - | - (pool positive rate 2.47%) | - |
| + top-100 | 0.480 | 355k | **1.85%** | 0.609 |
| + top-300 | 0.623 | 1.19M | **1.06%** | 0.658 |
| + top-1000 | **0.803** | 4.20M | **0.48%** | 0.736 |

Recall is recoverable to 80%. Precision is not: every added candidate set is
*worse* than the pool it joins, and the co-occurrence score only half-orders them.

**End-to-end test** (retrained reranker, pool + co-occ top-100, both arms scored
by the same cafaeval against the FULL ground truth so recall gain is credited and
precision cost penalised):

```
BASELINE (pool only)             f_micro_w = 0.1255
EXPANDED (pool + co-occ top-100) f_micro_w = 0.1276   delta = +0.0021
```

A 50% relative recall gain converts to **+0.0021**. The noise eats it.

Caveat, stated plainly: these absolutes are **not** comparable to the leaderboard's
0.218 (different harness, leaf-term ground truth, freshly retrained booster). The
*delta* is the measurement: same gt, same harness, expansion is the only variable.

## Conclusion: the wall is a RANKING limit

Three things are now measured, and they compose into one account:

1. **Not an evidence limit.** 97% of the true BP terms exist in the pre-t0
   vocabulary. The information is there.
2. **Not, primarily, a retrieval limit.** The pool we already retrieve (recall
   0.322) is worth **0.6077** to a perfect ranker; we deliver **0.1255**. Adding
   recall on top buys +0.0021 because retrieval is not what binds.
3. **A ranking limit.** No feature PROTEA carries exceeds AUC 0.68 on PK-BPO. Our
   signals (embedding-KNN, homology, taxonomy, self-prior, co-occurrence,
   text-alignment) are one flavour and mutually correlated; stacking them does not
   order this pool.

### What ranking quality would be needed, and what we tested for it

Calibrating AUC -> f_micro_w on this cell (blending the oracle with noise to hit
target AUCs, same harness):

| ranking | AUC | f_micro_w |
|---------|-----|-----------|
| random | 0.50 | 0.058 |
| | 0.658 | 0.114 |
| **our reranker** | **0.790** | **0.126** |
| | 0.834 | **0.238** |
| oracle | 1.00 | 0.43-0.61 |

The curve is steep exactly where we sit: **+0.044 AUC is worth +0.112 f_micro_w**.
So the target is modest in AUC terms: roughly **+0.04-0.05 AUC on PK-BPO**.

The natural hypothesis was "TransFew wins by modelling GO label structure; add that
as a signal". **We tested the two obvious forms of it and both are dead ends:**

- **co-occurrence** over the label set: association_cross AUC 0.60; as a candidate
  generator it bought +0.0021.
- **GO-DAG hierarchical proximity** (max Jaccard of ancestor sets between a candidate
  term and the protein's t0-known BP terms): **AUC 0.5501**, i.e. barely above chance.
  It is decorrelated from the reranker (corr -0.064) yet blending it in does not help
  (0.9/0.1 -> 0.7905, +0.0002; heavier blends make it worse).

So the claim "TransFew wins because of GO label structure" is a hypothesis this work
does **not** support: neither co-occurrence nor DAG proximity carries the missing
signal. Whatever TransFew exploits is presumably *learned* label representations
trained jointly with sequence, which is not reproducible as a bolt-on feature. The
honest statement is that we have **not identified** the signal that would buy the
missing 0.05 AUC, and the two structural candidates we could test are ruled out.

The upside of this framing: the headroom is real and it is inside the pool we
already build. Going from 20.7% capture to ~50% capture of that ceiling would clear
the +0.076 gap on PK-BPO several times over, and it needs only ~+0.05 AUC. The work
is a ranker, not a retriever.

The sobering half: we do not yet know what signal buys that AUC. Co-occurrence and
DAG proximity are ruled out by measurement. Until a candidate signal is found and
probed, "improve the ranker" is a direction, not a plan.

## What this means for the thesis

7/9 is the honest standing, and it is now a *characterized* result rather than a
shortfall: we state precisely where the loss is (ranking, quantified at 20.7% of the
achievable), that it is not an evidence ceiling, what mechanism would cross it, and
that the levers we currently hold are provably insufficient. NK-BPO #1 stands on its
own.

Method note, and a correction worth keeping: the first draft of this receipt read
the 0.322 recall as "recall-limited" and said so. The oracle test disproved it. A
recall number alone does not tell you what binds; the ceiling of the pool does.

Receipts: `SPLIT13_AUDIT.md`, `PATH_A_PROTST_EXECUTION.md`, `AB_RESULT.md`,
`storage/cooc_experiment/` (train_allfeat.py, fuse_and_score.py, cooc_fusion_result.json).
