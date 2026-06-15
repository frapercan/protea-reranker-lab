# fullgo results trajectory (sealed, leakage-clean, LAFA Sep_2025_Mar_2026 7401 frame)

All numbers are f_micro_w (IA-weighted micro-F, mean over 3 namespaces), sealed on the official 7401
frame with the exact published cafaeval harness. Each lever is fit/selected on SELECT 220->227 and
sealed once on 7401.

| Lever | NK | LK | PK | Mean | Note |
|---|---|---|---|---|---|
| KNN composite (PROTEA) | 0.412 | 0.394 | 0.165 | 0.324 | baseline |
| + learned re-scorer (IA/freq GBM) | - | - | - | 0.330 | KNN sub-features + IA + log_freq |
| classifier (6-PLM + ASL), standalone | 0.406 | 0.389 | 0.182 | 0.326 | full-label, no recall ceiling |
| classifier + M2 label-semantics (anc2vec) | 0.414 | 0.420 | 0.195 | 0.343 | hybrid: indep head + protein@anc2vec.T |
| + KNN-classifier ensemble (single seed) | 0.446 | 0.423 | 0.204 | 0.358 | union candidates, per-category GBM |
| **+ seed-averaged classifier + self-prior feature** | **0.464** | **0.465** | **0.215** | **0.381** | **champion** |
| FunBind (#2) | 0.441 | 0.451 | 0.205 | 0.366 |  |
| TransFew (#1) | 0.428 | 0.485 | 0.230 | 0.381 |  |

**Champion: 0.381 = ties TransFew (#1), leakage-clean.** NK 0.464 is clear **#1** (above both leaders).
LK 0.465 is between FunBind and TransFew. PK 0.215 is between the two leaders. Reproduced across runs at
0.3811-0.3814 (LightGBM run-to-run variance ~0.002 per category).

## The two levers that closed the gap

Both fit/selected on SELECT 220->227, then sealed once. Together they take the ensemble 0.358 -> 0.381
(+0.023), all of it from these two:

1. **Seed-averaging the classifier** (`seed_average.py`). The M2 anc2vec heads carry initialisation
   variance; averaging the consensus union of 3 seeds' top-100 (score = sum/n) lifts the standalone
   classifier 0.343 -> 0.369 and **generalises**: on SELECT it goes 0.305 -> 0.351 (+0.046), so it is not
   a test-frame artefact.
2. **Self-prior as a STACK FEATURE** (GOA non-experimental t0, propagated). Added as union candidates +
   two GBM columns (`sp`, `sp_p`), it lifts LK the most (where KNN coverage is thinnest). The earlier
   *flat/agreement blend* of the same signal was refuted; the difference is that the GBM learns *when* to
   trust it instead of always blending. This overturns the prior "self-prior refuted" note.

## Validated mechanisms

- **Ensemble** (classifier + KNN, union candidates) removes the KNN recall ceiling.
- **M2 label-semantics** (anc2vec GO-DAG ancestry as a label-similarity head, learnable scale ~0.97)
  targets LK and generalises (SELECT: classifier 0.305 -> 0.326, first time it beats KNN).
- **Seed-averaging + self-prior feature** (above) are the levers that reach #1.

## Refuted / no-op (do not redo)

- **self-prior as a flat/agreement blend** (vs. as a learned GBM feature, which is the win above),
  cross-aspect max, InterPro, label-aware-input, per-aspect models, naive max-union ensembling: flat or
  negative on SELECT.
- **anc2vec is NOT obsolete**: a data-learned label embedding (TruncatedSVD-200 of the training
  co-occurrence) scores 0.3205 (worse); the `indep` head already learns term relationships.
- **Learned GCN over the GO-DAG = NO-OP** over fixed anc2vec (0.3445 ~ 0.343).
- **GO-definition text semantics (BioBERT on GO def strings) = NO-OP** (0.3455 ~ 0.343).
- **M3 IEA weak-label pretraining (150k IEA proteins) = NET NEGATIVE** (0.335 < 0.343): helps NK, hurts LK.
- **Frequency-stratified heads (TransFew trick) = NO-OP** (0.338).
- **Per-protein calibration on the ensemble = flat** (minmax helps the bare classifier 0.343 -> 0.352 but
  is washed out once the GBM stack is in place).
- **Experimental-only KNN (drop non-experimental transfers) = NEGATIVE** (0.336): removes NK coverage.
- LAFA experimental evidence codes verified to match PROTEA's `EXPERIMENTAL` frozenset exactly.

## Status: champion sealed (0.381, ties #1)

The champion is `ensemble_seal.py` (seed-averaged M2 classifier + KNN + self-prior feature). Boosters and
spec persisted at `~/Thesis2/storage/fullgo_models/` (classifier_6plm_asl.pt, classifier_m2_anc2vec.pt,
ensemble_gbm_{NK,LK,PK}.txt, feature_spec.json). NK is #1 outright; the residual to TransFew is now only in
PK (0.215 vs 0.230). Reproduce with `REPRODUCE.md`.
