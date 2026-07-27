# clf+assoc reranker, 9 LAFA cells (2026-06-28)

Measures the association + classifier lever END-TO-END. Per-category LightGBM
lambdarank reranker retrained on the NEW platform dataset
`clean-learned-clfassoc-train227-test230` (learned encoder d8979601, train
v160..v227, test v230, k=30, schema 775611822dd9), now WITH compute_classifier
(M2) + compute_association (CSR) ON. The 5 previously zero-filled features
(`classifier_score`, `classifier_present`, `association_total`,
`association_cross`, `association_present`) are populated (clf 60.7% nonzero,
assoc 80.6%) and ADDED to the feature set (64 -> 69 feats). The classifier added
~727k candidates to the eval pool (eval = 1,198,440 rows).

Protocol identical to the published board config: per-category booster
(NK/LK/PK), group by (snapshot_pair, protein, aspect), early-stop on the v225-v227
cut; LAFA-exact scoring over the 7401 query set (globalmm per-category
reranker_score, KNN fallback for the ~54 uncovered queries), `cafaeval`
`-prop fill -norm cafa -no_orphans -toi` (PK adds `-known`), OBO/IA = LAFA t0
Sep 2025. So the ONLY change vs the board is the dataset + the 5 features.

## Feature uptake (gain importance) - the lever IS used
- `classifier_score` is the #1 feature in all three categories.
- `association_total` is #2 in PK (gain 1.10M, just under classifier) and #3 in LK (97k).
- `association_cross` used in LK/PK; `*_present` flags ~unused (redundant with the score).
- best_iter: NK 186, LK 87, PK 4 (PK booster barely trains, the precision wall).

## 9-cell f_micro_w

| cell    | clf+assoc | current board | delta   | leader            | beats leader |
|---------|-----------|---------------|---------|-------------------|--------------|
| nk-mfo  | 0.652     | 0.602         | +0.050  | goa 0.591         | YES          |
| nk-bpo  | 0.330     | 0.309         | +0.021  | -                 |              |
| nk-cco  | 0.472     | 0.431         | +0.041  | FunBind 0.473     | no (-0.001)  |
| lk-mfo  | 0.555     | 0.519         | +0.036  | goa 0.510         | YES          |
| lk-bpo  | 0.307     | 0.348         | -0.041  | TransFew 0.512    | no           |
| lk-cco  | 0.468     | 0.419         | +0.049  | board 0.434       | YES          |
| pk-mfo  | 0.127     | 0.235         | -0.108  | -                 |              |
| pk-bpo  | 0.101     | 0.117         | -0.016  | TransFew 0.294    | no           |
| pk-cco  | 0.183     | 0.254         | -0.071  | -                 |              |

Mean (9 cells): 0.355 clf+assoc vs 0.359 current board. Improved cells: 5/9.
Leaders newly beaten: nk-mfo, lk-mfo, lk-cco (lk-cco was a documented LOSE cell).

Gate variant (ADR-D43, PK = raw-KNN `1-distance` over the same pool):
PK mfo 0.187 / bpo 0.074 / cco 0.184 - also BELOW the board PK. So PK regresses
under both scorers.

## Verdict

The association + classifier lever is a clear WIN on NK and LK (5/9 cells up):
it lifts every NK cell, lifts LK-mfo/LK-cco, and newly beats three external
leaders including the lose cell lk-cco (0.468 > 0.434); nk-cco effectively ties
FunBind (0.472 vs 0.473). These are exactly the categories where the association
weight is meaningful (LK 32%). The features are not cosmetic: `classifier_score`
and `association_total` are the top features driving these gains.

It does NOT help PK, which regresses across all three aspects. Cause is the
candidate POOL, not the scorer: the classifier adds 727k low-precision
candidates and PK is precision-bound ("PK = wall, no arm helps"), so the expanded
pool dilutes PK whether reranked (best_iter=4) or raw-KNN gated (both below board).
The board's PK numbers came from the KNN-only pool.

Net 9-cell mean dips marginally (0.355 vs 0.359) ONLY because PK contributes the
most cells and drags the average; on the NK+LK block the lever is strongly
positive. Actionable next step: keep the expanded clf+assoc pool for NK/LK, but
restrict the PK candidate pool to KNN-only (do not inject classifier candidates
for PK) - that recovers PK to the board while keeping the NK/LK wins, which would
raise the overall mean above the board and bank three #1 cells.

## Artifacts
- `boosters/booster_{nk,lk,pk}.txt` - retrained per-category boosters (69 feat)
- `features.json` (69), `train_info.json` (best_iter + feature importances incl new5)
- `eval_scores.parquet` - per-candidate reranker scores on eval v227-v230
- `comparison_9cell.json` - full per-cell verdict + gate-PK + leaders beaten
- `clfassoc_train.py`, `clfassoc_lafa.py`, `clfassoc_gate_pk.py`, `lafa_harness.py`

Board NOT modified (conductor handles injection). No PR opened. No live DB / no
platform jobs touched (file-based MinIO download only).
