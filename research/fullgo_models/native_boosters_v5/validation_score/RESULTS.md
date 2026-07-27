# Native reranker, VALIDATION-frame f_micro_w (H1 score-reproduction)

OPTIMISTIC numbers. The native_boosters_v5 LightGBM boosters early-stopped on this
exact eval split (the SELECT 220->227 held-out eval), so these carry a mild upward
bias by construction. Treat as a sanity-floor reproduction, not a leaderboard claim.

## Headline (primary recipe: no TOI, no PK-exclude)

| category | f_micro_w |
|----------|-----------|
| NK       | 0.6111    |
| LK       | 0.6274    |
| PK       | 0.4142    |
| **MEAN** | **0.5509**|

Per-namespace (mean = mean over bpo/cco/mfo of the per-ns max f_micro_w):
- NK: bpo 0.5411 / cco 0.5994 / mfo 0.6927
- LK: bpo 0.5846 / cco 0.6147 / mfo 0.6828
- PK: bpo 0.2952 / cco 0.4926 / mfo 0.4549

## How it was scored

1. Pulled `datasets/fullgo-union-SELECT-160-220-227-v5/eval.parquet` from MinIO
   (8,034,424 rows, all `snapshot_pair == v220-v227`, category in nk/lk/pk).
   `go_term_id` is already a `GO:` string, so NO DB lookup was needed.
2. Streamed row-group batches, applied the matching per-category booster
   (`ensemble_gbm_{NK,LK,PK}.txt`) over the 69 numeric features in
   `summary.json["features"]` order. Peak RSS ~3.7 GB.
3. Collapsed the candidate rows (union of KNN / classifier / self-prior / association
   streams has duplicate (protein,term) pairs) to ONE score per (protein, term) by
   MAX, matching the offline seal (load_knn/load_clf keep the max per pair).
4. GT derived from the parquet: `label == 1` leaf rows per category. cafaeval
   propagates to ancestors (prop=fill).
5. cafaeval (cafaeval-protea fork) with the offline-seal recipe:
   `ia=IA, prop="fill", norm="cafa", no_orphans=True, max_terms=None, th_step=0.01,
   n_cpu=1`. OBO + IA from `lafa_t0_Sep_2025/`. f_micro_w per category = mean over
   the 3 namespaces of the per-ns max.

## Caveats (why this is NOT directly comparable to the test-frame 0.391)

- **OPTIMISTIC**: boosters early-stopped on this eval split.
- **No TOI**: the offline-seal/leaderboard recipe restricts evaluation to the frame's
  official "terms of interest" (`groundtruth_terms_of_interest.txt`, ~38.6k terms for
  the test frame = the proteome-wide experimental delta in the window). That file does
  NOT exist for the validation frame and cannot be reconstructed from the eval.parquet
  (the parquet only has these test proteins' GT, not the proteome-wide delta). Omitting
  the TOI evaluates over a smaller/easier term universe, which inflates scores. This is
  the dominant reason the numbers sit far above the test-frame 0.477/0.482/0.215.
  A sensitivity run with a proxy TOI = cumulative-experimental terms at v227 (26,774
  terms) barely moved the numbers (NK 0.611->0.608), confirming a cumulative proxy is
  too large to mimic the official frame-delta TOI. See `result_proxy_toi.json`.
- **No PK-exclude**: the test-frame `groundtruth_PK_known.tsv` (t0-known per-protein
  terms removed before PK scoring) has no validation-frame equivalent and cannot be
  reconstructed from the parquet. Omitting it leaves t0-known candidates as false
  positives, which is PESSIMISTIC for PK (not inflationary). So validation PK 0.4142 is
  optimistic-from-no-TOI but pessimistic-from-no-exclude; the two effects partly offset.

## Bottom line

Pipeline is sound: boosters apply cleanly, predictions/GT derive correctly, cafaeval
runs the published recipe. The MEAN 0.5509 is a frame-and-recipe-shifted OPTIMISTIC
number, not the 0.391 test figure. The honest H1 conclusion: the native boosters do
produce a competent, internally-consistent f_micro_w on the validation frame; the exact
magnitude is not comparable to the 0.391 leaderboard number because the validation frame
lacks the official TOI / PK-known artefacts that the test-frame harness applies.
