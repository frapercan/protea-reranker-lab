# ProtST reranker A/B - 2-seed directional robustness (sealed 227->230 frame)

Directional cross-check of the ProtST reranker A/B on TWO LightGBM seeds {42, 7}. armA =
champion baseline (3 protst columns excluded); armB = champion + protst. Same enriched
parquets (`enriched_train.parquet` / `enriched_eval.parquet`, NO re-enrichment), same harness
(`train_ab_seed.py`, `score_ab.py`), same PARAMS; the only per-run change is the LightGBM
master `seed`, which cascades to bagging_seed / feature_fraction_seed / data_random_seed
(verified: same seed -> identical predictions, different seed -> divergent). seed42 reuses the
existing `armA_9cell.json` / `armB_9cell.json`; seed7 is the new pair. This is a directional
seed check only; the rigorous significance evidence is the protein-level bootstrap CIs on the
BP cells (which exclude zero), not this table.

## 9-cell + mean9 f_micro_w, both seeds

delta = armB - armA. sign = agreement of the two per-seed deltas.

| cell    | s42 armA | s42 armB | s42 delta | s7 armA | s7 armB | s7 delta | sign |
|---------|----------|----------|-----------|---------|---------|----------|------|
| nk-mfo  | 0.6367 | 0.6249 | -0.0118 | 0.6245 | 0.6353 | +0.0108 | FLIP |
| nk-bpo  | 0.4607 | 0.4911 | +0.0304 | 0.4669 | 0.4790 | +0.0121 | both + |
| nk-cco  | 0.5488 | 0.5380 | -0.0108 | 0.5378 | 0.5395 | +0.0017 | FLIP |
| lk-mfo  | 0.5570 | 0.5483 | -0.0087 | 0.5534 | 0.5359 | -0.0175 | both - |
| lk-bpo  | 0.4901 | 0.5093 | +0.0192 | 0.4803 | 0.5138 | +0.0335 | both + |
| lk-cco  | 0.5054 | 0.5364 | +0.0310 | 0.5108 | 0.5520 | +0.0412 | both + |
| pk-mfo  | 0.4707 | 0.4704 | -0.0003 | 0.4694 | 0.4701 | +0.0007 | FLIP |
| pk-bpo  | 0.3433 | 0.3525 | +0.0092 | 0.3488 | 0.3547 | +0.0059 | both + |
| pk-cco  | 0.4886 | 0.5093 | +0.0207 | 0.4829 | 0.5009 | +0.0180 | both + |
| mean9   | 0.5001 | 0.5089 | +0.0088 | 0.4972 | 0.5090 | +0.0118 | both + |

## Verdict

- (a) BP wall (nk/lk/pk-bpo): POSITIVE IN BOTH SEEDS for all three (lk-bpo +0.0192/+0.0335, pk-bpo +0.0092/+0.0059, nk-bpo +0.0304/+0.0121); the BP lift is real and sign-stable.
- (b) MF/CC dips are MIXED: consistently negative (likely real): lk-mfo -0.0087/-0.0175; sign-flips (likely noise): nk-mfo -0.0118/+0.0108, nk-cco -0.0108/+0.0017
- (c) mean9 delta: seed42 +0.0088, seed7 +0.0118 (net positive in both seeds).

## Non-committal notes

- Offline A/B delta vs OUR champion only. Nothing here touches the served reranker, promotes,
  or runs the platform eval. Informs the global-vs-BP-gated injection decision, nothing more.
- A single seed knob controls all LightGBM RNG (verified), so both seeds sit on identical footing.
