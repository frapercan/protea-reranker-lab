# fullgo — full-label GO classifier + KNN ensemble (LAFA #1 push)

A learned, full-label GO term classifier over frozen multi-PLM embeddings, ensembled with the
PROTEA KNN composite. This is the durable home of the work prototyped during the 2026-06-14 #1 push
(previously scattered in `/tmp`). It supersedes the abandoned per-pair reranker scaffold
(`protea-neural-head`), which inherited the KNN recall ceiling and is not full-label.

## Result (sealed, leakage-clean)

On the official LAFA `Sep_2025_Mar_2026` frame (7401 targets, official groundtruth, exact published
cafaeval harness), f_micro_w (IA-weighted micro-F, mean over the three namespaces):

| Method | NK | LK | PK | Mean |
|---|---|---|---|---|
| KNN composite (PROTEA) | 0.412 | 0.394 | 0.165 | 0.324 |
| full-label classifier (6-PLM + ASL) | 0.406 | 0.389 | 0.182 | 0.326 |
| **KNN + classifier learned ensemble** | **0.447** | **0.402** | **0.199** | **0.349** |
| FunBind (#2) | 0.441 | 0.451 | 0.205 | 0.366 |
| TransFew (#1) | 0.428 | 0.485 | 0.230 | 0.381 |

Ensemble = **#3**, NK already **#1** (above FunBind). Validated two ways: the classifier recipe
generalises (trained on v220, evaluated on the SELECT 220->227 frame: mean 0.305, beats KNN on NK),
so the 7401 number is not test-overfit.

## Recipe (FROZEN; change only via SELECT validation)

- **Classifier:** concat 6 frozen pooled PLM embeddings (Ankh-base 768 + ESM2-3B 2560 + Ankh-large 1536
  + ESM2-650M 1280 + ESMC-600M 1152 + ProtT5 1024 = 8320-d) -> 2x1024 MLP (LayerNorm+GELU+Dropout 0.2)
  -> |vocab| logits. **Asymmetric Loss (ASL)** gamma_neg=4, gamma_pos=1, clip=0.05. 30 epochs, AdamW
  lr 1e-3. Labels = propagated t0 experimental annotations restricted to the benchmark terms-of-interest
  vocab. Per-protein top-100 predictions.
- **Ensemble:** per-category LightGBM over features [KNN composite score + KNN sub-features (distance,
  identity_nw/sw, taxonomic_distance, neighbor_vote_fraction) + classifier_score + knn_present +
  clf_present + term IA + log t0-pool-frequency]. **Candidates = union(KNN terms, classifier terms)** ->
  removes the KNN recall ceiling. Fit per category on SELECT 220->227, sealed once on 7401.
- **Frozen embeddings only** (no PLM fine-tuning; fits a single 12GB GPU).

## Discipline (non-negotiable)

- Strict temporal cutoff: labels <= t0. Train on t0, validate on SELECT 220->227, **seal once** on 7401.
- Evaluate ONLY with the exact published harness (`cafaeval -ia IA.tsv -toi terms_of_interest
  -known(exclude, propagated) -no_orphans -prop fill -norm cafa`, Sep_2025 OBO). Never an internal grid.
- Beat-or-revert at every change. The 7401 number is a faithful LAFA-frame score but NOT an official
  leaderboard entry; a real position requires a submission container (see `protea-lafa-knn`).

## Pipeline

```
extract.py        # per-protein 6-PLM concat embeddings + propagated t0 labels  -> data .npz
train_classifier.py  # 6-PLM + ASL full-label classifier                         -> predictions .tsv
ensemble.py       # per-category GBM over KNN + classifier features, sealed       -> sealed metrics
evaluate.py       # exact published cafaeval harness on the 7401 frame
```

Config (PLM config ids, ASL params, frames, DB) in `config.yaml`. See `REPRODUCE.md` for the exact
commands and the SELECT/TEST frame definitions.

## Next (toward #1, frozen-embedding, on-box)

- **M2 label semantics** (LK gap): GO-definition text embeddings + a GCN over the GO-DAG as a label
  encoder (TransFew's mechanism: predict via GO-embedding similarity, transfer to related terms).
- **M3 scale** (PK): train on the full annotated proteome (556k) with IEA weak-label pretraining,
  strict <=t0; frequency-grouped heads for rare terms.
- Refuted, do NOT redo: self-prior (max/agreement), cross-aspect, InterPro, label-aware-input,
  per-aspect models, naive max-union ensembling.
