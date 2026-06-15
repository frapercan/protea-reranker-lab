# fullgo results trajectory (sealed, leakage-clean, LAFA Sep_2025_Mar_2026 7401 frame)

All numbers are f_micro_w (IA-weighted micro-F, mean over 3 namespaces), sealed on the official 7401
frame with the exact published cafaeval harness. Each lever is fit/selected on SELECT 220->227 and
sealed once on 7401.

| Lever | NK | LK | PK | Mean | Note |
|---|---|---|---|---|---|
| KNN composite (PROTEA) | 0.412 | 0.394 | 0.165 | 0.324 | baseline |
| + learned re-scorer (IA/freq GBM) | - | - | - | 0.330 | KNN sub-features + IA + log_freq |
| classifier (6-PLM + ASL), standalone | 0.406 | 0.389 | 0.182 | 0.326 | full-label, no recall ceiling |
| + KNN-classifier learned ensemble | 0.447 | 0.402 | 0.199 | 0.349 | union candidates, per-category GBM |
| classifier + M2 label-semantics (anc2vec) | 0.414 | 0.420 | 0.195 | 0.343 | hybrid: indep head + protein@anc2vec.T |
| **+ ensemble (with M2 classifier)** | **0.448** | **0.413** | **0.205** | **0.355** | **current best** |
| FunBind (#2) | 0.441 | 0.451 | 0.205 | 0.366 | gap +0.011 |
| TransFew (#1) | 0.428 | 0.485 | 0.230 | 0.381 | gap +0.026 |

**Current: 0.355 = #3.** NK 0.448 is **#1** (above FunBind). PK 0.205 **ties** FunBind. The gap to #1 is
+0.026, concentrated in **LK (0.413 vs 0.485)**.

## Validated mechanisms

- **Ensemble** (classifier + KNN, union candidates) removes the KNN recall ceiling: +0.023 over KNN.
- **M2 label-semantics** (anc2vec GO-DAG ancestry embeddings as a label-similarity head, learnable
  scale ~0.97) targets LK and generalises (SELECT: classifier 0.305 -> 0.326, first time it beats KNN).
  +0.006 to the ensemble.

## Refuted / no-op (do not redo)

- self-prior (max/agreement blend), cross-aspect (max + learned), InterPro, label-aware-input,
  per-aspect models, naive max-union ensembling: all flat or negative on SELECT.
- **Learned GCN over the GO-DAG = NO-OP over fixed anc2vec** (0.3445 ~ 0.343): anc2vec already encodes
  DAG ancestry; re-propagating it adds nothing.
- **GO-definition text semantics (BioBERT on GO def strings, concat with anc2vec) = NO-OP** (0.3455 ~ 0.343,
  LK slightly down): label semantics are SATURATED at anc2vec; the BioBERT def-text adds no separable signal.
- **M3 IEA weak-label pretraining (150k IEA proteins) = NET NEGATIVE** (mean 0.335 < M2 0.343): IEA labels
  are electronic/homology -> help NK (+0.022) but HURT LK (-0.041, our gap); the fine-tune does not recover.

## Status: practical optimum reached (0.355)

All frozen-embedding levers are exhausted. The ensemble (+0.023) and M2-anc2vec (+0.006) are the only wins;
everything else is no-op, net-negative, or refuted. The remaining gap to #1 (TransFew 0.381, +0.026) is in
**LK (0.413 vs 0.485)** and is NOT closable by better label embeddings, self-prior, or IEA scale. Crossing
it would require a different model class -- PLM fine-tuning (ruled out by the 12 GB GPU) or a multimodal
structure model (like FunBind) -- i.e. a new project, not another lever here.

**Recommendation: 0.355 is the sealed, validated, citable result (#3, NK #1, PK ties FunBind).**

Weights persisted at `~/Thesis2/storage/fullgo_models/` (classifier_6plm_asl.pt, classifier_m2_anc2vec.pt,
ensemble_gbm_{NK,LK,PK}.txt).
