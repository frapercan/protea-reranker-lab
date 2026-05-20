# binary-objective per-cell recipe (study 27): publishable CI summary

Study: study-27 (binary-objective per-cell recipe, multiseed).
Hparams: binary objective, lean+lin+emb (56 features), neg_pos_ratio=10,
  num_boost_round=10000, lr=0.05, num_leaves=63, min_data_in_leaf=100,
  early_stopping_rounds=100.
Seeds: 42, 137, 244 (3-seed replication of the study-26 binary champion).
Eval set: bench-v1-K5-v226-lineage-prostt5 (eval window v226-v230).
Cafaeval: prop=fill, norm=cafa, no_orphans=True.

## Per-cell cafaeval Fmax: binary recipe study 27 (mean +- 95% CI half-width)

| cell | seed=42 | seed=137 | seed=244 | mean | CI half-width |
|-|-|-|-|-|-|
| nk-mfo | 0.7376 | 0.7436 | 0.7411 | 0.7408 | 0.0030 |
| nk-bpo | 0.5848 | 0.5918 | 0.5895 | 0.5887 | 0.0035 |
| nk-cco | 0.7992 | 0.7999 | 0.7949 | 0.7980 | 0.0025 |
| lk-mfo | 0.6833 | 0.6820 | 0.6809 | 0.6821 | 0.0012 |
| lk-bpo | 0.6629 | 0.6680 | 0.6724 | 0.6678 | 0.0048 |
| lk-cco | 0.7948 | 0.8001 | 0.7971 | 0.7973 | 0.0027 |

## binary recipe (study 27) vs lambdarank (study 22, LB.2) comparison

lambdarank (study 22) seeds: 42, 7, 137 (LB.2, leakage-fixed lambdarank, lean+lin features).
binary (study 27) seeds: 42, 137, 244 (this run, binary objective, lean+lin+emb features).
Bootstrap: N=10000, independent arms (different seed sets).
sig_95: 1 if delta 95% CI lower bound > 0.

| cell | binary (study 27) mean | lambdarank (study 22) mean | delta | delta 95% CI | sig_95 |
|-|-|-|-|-|-|
| nk-mfo | 0.7408 | 0.7065 | +0.0343 | [+0.0296, +0.0387] | 1 |
| nk-bpo | 0.5887 | 0.5596 | +0.0291 | [+0.0252, +0.0330] | 1 |
| nk-cco | 0.7980 | 0.7774 | +0.0206 | [+0.0152, +0.0255] | 1 |
| lk-mfo | 0.6821 | 0.6807 | +0.0014 | [-0.0052, +0.0063] | 0 |
| lk-bpo | 0.6678 | 0.6459 | +0.0218 | [+0.0165, +0.0272] | 1 |
| lk-cco | 0.7973 | 0.7368 | +0.0604 | [+0.0527, +0.0711] | 1 |

## binary single-seed (study 26) vs binary multiseed (study 27) comparison

| cell | study 26 (seed=42) | study 27 mean | delta |
|-|-|-|-|
| nk-mfo | 0.7376 | 0.7408 | +0.0032 |
| nk-bpo | 0.5848 | 0.5887 | +0.0039 |
| nk-cco | 0.7992 | 0.7980 | -0.0012 |
| lk-mfo | 0.6915 | 0.6821 | -0.0094 |
| lk-bpo | 0.6639 | 0.6678 | +0.0039 |
| lk-cco | 0.7948 | 0.7973 | +0.0025 |

## Training wall-clock (per seed)

| cell | seed=42 (min) | seed=137 (min) | seed=244 (min) |
|-|-|-|-|
| nk-mfo | 0.1 | 0.1 | 0.1 |
| nk-bpo | 0.2 | 0.2 | 0.2 |
| nk-cco | 0.1 | 0.1 | 0.1 |
| lk-mfo | 0.1 | 0.1 | 0.1 |
| lk-bpo | 0.1 | 0.1 | 0.1 |
| lk-cco | 0.1 | 0.1 | 0.1 |

## Outcome

See the per-cell CI table above for the full multiseed results.
Artifact root: runs/v27_binary_multiseed/
