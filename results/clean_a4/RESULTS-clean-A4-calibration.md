# Track C2: does per-aspect calibration recover the PK regression?

C1 found that the per-category LambdaMART reranker lifts NK strongly but
regresses PK (a precision collapse). The hypothesis for C2 was that lambdarank
emits an uncalibrated score and cafaeval sweeps a fixed threshold grid
(`np.arange(0.001, 1, 0.001)`), so an out-of-range score cannot be thresholded
and precision is capped. C2 fits per (category, aspect) calibrators (isotonic and
Platt) on the v220-v225 holdout (NOT the eval frame), maps the reranker score
into [0, 1], and re-evaluates on the validation frame v225 -> v227.

Provenance is identical to C1 (dataset `fullgo-clean-A4-train225-val`, schema
`775611822dd9`, train up to v220-v225, eval v225-v227, band v227,
`association_*` / `classifier_score` zero-filled and dropped). The numpy
bootstrap was switched to cafaeval's fixed grid so its sweep is scale-faithful.
cafaeval is authoritative for every point f_micro_w; the bootstrap CI is the
paired uncertainty (its absolute level offsets cancel in the paired delta).

## Per-cell f_micro_w: raw-KNN vs reranked (uncal / isotonic / Platt)

The uncalibrated column reproduces C1 exactly (cross-check passed, e.g. pk-bpo
0.2738).

| cell   |   raw  | uncal  | isotonic | Platt  | delta iso vs raw | iso CI (boot) | p(<=0) |
|--------|-------:|-------:|---------:|-------:|-----------------:|---------------|-------:|
| nk-mfo | 0.5016 | 0.5645 | 0.5376 | 0.5400 | +0.0360 | [+0.015, +0.052] | 0.00 |
| nk-bpo | 0.4253 | 0.4713 | 0.4526 | 0.4552 | +0.0273 | [-0.004, +0.041] | 0.05 |
| nk-cco | 0.4516 | 0.5545 | 0.5336 | 0.5366 | +0.0820 | [+0.055, +0.110] | 0.00 |
| lk-mfo | 0.4705 | 0.5089 | 0.4966 | 0.5019 | +0.0261 | [+0.001, +0.053] | 0.02 |
| lk-bpo | 0.5094 | 0.5190 | 0.5028 | 0.5026 | -0.0066 | [-0.024, +0.003] | 0.93 |
| lk-cco | 0.4562 | 0.4807 | 0.5006 | 0.5094 | +0.0444 | [+0.011, +0.075] | 0.01 |
| pk-mfo | 0.4077 | 0.3805 | 0.3713 | 0.3711 | -0.0364 | [-0.059, -0.031] | 1.00 |
| pk-bpo | 0.4463 | 0.2738 | 0.2687 | 0.2579 | -0.1776 | [-0.168, -0.149] | 1.00 |
| pk-cco | 0.4080 | 0.3648 | 0.3577 | 0.3556 | -0.0503 | [-0.088, -0.059] | 1.00 |

Per-cell mean delta vs raw-KNN: uncalibrated **+0.0046**, isotonic **-0.0061**,
Platt similar to isotonic. Calibration does not lift the mean; it slightly lowers
it (the NK/LK gains shrink, except lk-cco where isotonic helps).

## Q1: does calibration recover PK? NO.

PK isotonic and Platt are essentially identical to uncalibrated, and all three
regress significantly (pk-bpo -0.178, pk-mfo -0.036, pk-cco -0.050; every CI
excludes 0, p=1.00). The PK precision wall does not move:

| cell   | raw P / R       | uncal P / R     | isotonic P / R  |
|--------|-----------------|-----------------|-----------------|
| pk-mfo | 0.213 / 0.515 | 0.159 / 0.677 | 0.157 / 0.710 |
| pk-bpo | 0.363 / 0.472 | 0.167 / 0.513 | 0.168 / 0.509 |
| pk-cco | 0.186 / 0.419 | 0.114 / 0.564 | 0.108 / 0.635 |

IA-weighted precision is unchanged by calibration (pk-bpo precision 0.167 uncal
vs 0.168 isotonic). This shows the PK regression is NOT a score-scale artifact:
it is intrinsic to the reranker's within-cell pooled ordering, where lambdarank
trades precision for recall. A single monotonic per-cell remap cannot fix a
cross-protein ordering, so calibration is the wrong tool for PK.

## Q2: with calibration, do we beat the champion across ALL categories?

Two framings:

Per-cell-optimal threshold (the C1 framing, each cell scored independently):

| config | mean f_micro_w delta vs champion |
|--------|---------------------------------:|
| reranked-all, uncalibrated | +0.0046 |
| reranked-all, isotonic | -0.0061 |
| per-category GATE (NK/LK reranked, PK raw-KNN), isotonic | +0.0240 |

The gate still beats the champion, but calibration shaves it from the C1
uncalibrated gate (+0.031) down to +0.024. Reranked-all never beats the champion
overall: PK drags the mean to roughly flat (or negative under calibration).

Pooled per-aspect (CAFA-style: NK+LK+PK under one shared threshold):

| aspect | champion (raw-KNN) | reranked-all isotonic | gated isotonic |
|--------|-------------------:|----------------------:|---------------:|
| mfo | 0.4376 | 0.3519 | 0.3514 |
| bpo | 0.4529 | 0.2168 | 0.2168 |
| cco | 0.4252 | 0.3511 | 0.4176 |
| MEAN | **0.4386** | 0.3066 | 0.3286 |

Under realistic pooled scoring the champion (raw-KNN) wins decisively. PK
dominates the pooled set by protein count (pk-bpo has 8,639 proteins vs 811 NK,
1,640 LK), so the PK regression sinks the pooled metric even with the gate.
Important honest nuance: the C1 "+0.031 gate beats champion" result holds only
under the per-cell-optimal-threshold framing; it does not survive a single shared
threshold per aspect.

## Verdict

Calibration is NOT the PK lever. Isotonic and Platt leave the PK regression and
its precision collapse essentially unchanged, and slightly reduce the NK/LK gains
per cell. The honest conclusions stand:

- The per-category GATE (apply the reranker on NK and LK, keep raw-KNN on PK) is
  the best config, and only under per-cell thresholds.
- PK needs a better model, not a rescaling. The most direct lever is a v2 export
  with the association (cooccurrence) and classifier compute flags ON: those two
  evidence ports are zero-filled here and are exactly the signals that
  historically carried PK precision. Retraining the PK arm with a precision-
  oriented objective (or a binary objective with a precision-tuned threshold) is
  the secondary lever to test once those features exist.

Reproduce: `scripts/run_clean_a4_calibration.py`; record in
`results/clean_a4/calibration_results.json`.
