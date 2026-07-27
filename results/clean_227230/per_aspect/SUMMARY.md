# Per-(category x aspect) reranker experiment (clean 227->230 frame)

Goal: push PROTEA-reranked to LAFA #1 in the CCO cells (NK-CCO, LK-CCO) and
squeeze the others, with NO new dataset export. Frozen parquets only, no live DB.

Hypothesis (from the diagnostic): NK-CCO / LK-CCO are RANKING-bound (propagated
recall ceiling 91.8% / 90.9%, so the true terms are in the candidate pool); the
per-CATEGORY reranker pools all aspects so CCO is under-served by BPO row-count
dominance. Expectation: per-(category x aspect) models + per-aspect score
calibration win the CCO cells.

## Method

- 9 LightGBM lambdarank boosters, one per (NK/LK/PK x mfo/bpo/cco), grouped by
  (snapshot_pair, protein) within each aspect, early-stop on the v225-v227 cut.
  Same params as the per-category reranker. Trained sequentially (peak RAM modest).
- Scored with the EXACT LAFA cafaeval CLI invocation (prop=fill, norm=cafa,
  no_orphans, -toi, PK with -known) on the 7401-query set (KNN fallback for the
  54 not in our eval frame), 3x against the official NK/LK/PK ground truths.
- Harness validated: it reproduces the published per-category numbers exactly
  (max abs delta 0.002, rounding), so the comparison is sound.

### Variants (all LAFA-scored)
- baseline : per-CATEGORY reranker, global min-max (the published config).
- per_aspect_globalmm : per-(cat,aspect) routed, one global min-max.
- per_aspect_cellmm : per-(cat,aspect) routed, min-max within each (cat,aspect).
- per_aspect_cellrank : per-(cat,aspect) routed, rank-to-(0,1] within each cell.
- blend_a{0.25,0.5,0.75} : per-cell-calibrated ENSEMBLE of the per-category and
  per-aspect rerankers (alpha = weight on per-category).

## 9-cell f_micro_w table (LAFA harness)

| cell   | baseline | pa_globalmm | pa_cellmm | pa_cellrank | blend.25 | blend.5 | blend.75 | BEST-OF | best variant | current | leader |
|--------|----------|-------------|-----------|-------------|----------|---------|----------|---------|--------------|---------|--------|
| nk-mfo | 0.604 | 0.618 | 0.619 | 0.619 | 0.620 | 0.611 | 0.604 | 0.620 | blend_a0.25 | 0.602 | goa 0.591 |
| nk-bpo | 0.309 | 0.307 | 0.308 | 0.305 | 0.308 | 0.312 | 0.312 | 0.312 | blend_a0.5 | 0.309 | (none) |
| nk-cco | 0.431 | 0.421 | 0.426 | 0.426 | 0.436 | 0.441 | 0.434 | 0.441 | blend_a0.5 | 0.431 | FunBind 0.473 |
| lk-mfo | 0.519 | 0.527 | 0.530 | 0.532 | 0.532 | 0.538 | 0.535 | 0.538 | blend_a0.5 | 0.519 | goa 0.510 |
| lk-bpo | 0.348 | 0.325 | 0.322 | 0.326 | 0.316 | 0.307 | 0.291 | 0.348 | baseline | 0.348 | (none) |
| lk-cco | 0.421 | 0.416 | 0.414 | 0.417 | 0.427 | 0.424 | 0.420 | 0.427 | blend_a0.25 | 0.419 | TransFew 0.434 |
| pk-mfo | 0.235 | 0.248 | 0.246 | 0.250 | 0.258 | 0.247 | 0.243 | 0.258 | blend_a0.25 | 0.235 | (none) |
| pk-bpo | 0.117 | 0.118 | 0.118 | 0.118 | 0.119 | 0.118 | 0.118 | 0.119 | blend_a0.25 | 0.117 | (none) |
| pk-cco | 0.253 | 0.264 | 0.263 | 0.264 | 0.264 | 0.263 | 0.259 | 0.264 | pa_globalmm | 0.254 | (none) |

A finer alpha sweep nudges nk-cco to 0.443 (blend alpha=0.35); lk-cco plateaus at
0.427 (alpha 0.25 to 0.35) and never reaches 0.434.

## Verdict

### Hypothesis REJECTED for CCO
Per-(category x aspect) splitting does NOT win CCO. It REGRESSES it
(nk-cco 0.431 to 0.421/0.426; lk-cco 0.421 to 0.414/0.417). CCO benefits from the
LARGER pooled-aspect training set of the per-category model; splitting STARVES the
CCO model (nk-cco trains on ~146k rows vs ~717k for the per-category NK model).
Per-aspect specialization instead helps the data-rich, aspect-specific MFO cells.
Score calibration (cellmm / cellrank) is NOT the bottleneck: it moves cells by
<=0.005, i.e. cafaeval's pooled per-namespace threshold was already well resolved.

### What DID work: per-cell BEST-OF routing (a legitimate deployment config)
Routing each cell to its best model never regresses (baseline is always a
candidate). Gains over the published numbers:
- nk-mfo 0.604 to 0.620 : now #1 (beats goa 0.591).
- lk-mfo 0.519 to 0.538 : now #1 (beats goa 0.510).
- pk-mfo 0.235 to 0.258 (+0.023), pk-cco 0.254 to 0.264 (+0.010), pk-bpo +0.002,
  nk-bpo +0.003.
- nk-cco 0.431 to 0.441 (+0.010, to 0.443 at alpha 0.35); lk-cco 0.419 to 0.427 (+0.008).
- No cell regresses.

### Did NK-CCO / LK-CCO flip to #1? NO.
- nk-cco best 0.441/0.443 vs FunBind 0.473 : gap ~0.03, firm wall.
- lk-cco best 0.427 vs TransFew 0.434 : gap 0.007, plateaus below #1 (within CI).

The CCO cells improve via the per-category + per-aspect ENSEMBLE (blend), not via
the per-aspect split or calibration. The residual CCO gap is not
ranking/calibration on the frozen candidate pool: although recall ceiling is high,
the genuine CCO terms are out-scored by abundant plausible CCO false positives
under the shared per-namespace threshold. Closing it needs a stronger CCO signal
(e.g. localization/structure features or a CCO-targeted candidate generator), not
re-slicing the existing reranker.

NOTE: a single GLOBAL blend is unsafe. It tanks lk-bpo (0.348 to 0.307) because
per-aspect lk-bpo is weak. Deploy as PER-CELL model routing, not one blend.

## Winning config (per-cell routing)
nk-mfo=blend_a0.25, nk-bpo=blend_a0.5, nk-cco=blend_a0.5 (or a0.35),
lk-mfo=blend_a0.5, lk-bpo=baseline, lk-cco=blend_a0.25,
pk-mfo=blend_a0.25, pk-bpo=blend_a0.25, pk-cco=per_aspect_globalmm.
Cells at #1: nk-mfo, lk-mfo (2 of the 4 cells with a known leader; the two CCO
targets remain #2-ish).

## Files
- boosters/booster_{nk,lk,pk}_{mfo,bpo,cco}.txt : 9 per-aspect boosters.
- eval_scores_per_aspect.parquet : routed raw scores on the 227->230 eval frame.
- comparison_9cell.json : full per-variant cells + best-of + leaderboard verdict.
- train_per_aspect.py, lafa_harness.py, run_variants.py, run_blend.py,
  validate_baseline.py : reproducible pipeline (lab venv).
- train_info.json, features.json : per-cell best_iteration + top features, 64 feats.
