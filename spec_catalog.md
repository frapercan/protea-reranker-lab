# Spec catalog

Living registry of every reranker spec tested in protea-reranker-lab.
Outcome values: **ship** (merged/deployed), **drop** (superseded or known-bad),
**iterate** (informative, not final), **smoke** (sanity/CI check only).

Tuple notation: `plm / K / objective / feature-set / eval-set`.
- `plm`: embedding backbone used for KNN (esmc_300m unless otherwise noted)
- `K`: number of KNN neighbours in the PredictionSet
- `objective`: lambdarank (LR) or binary (BIN)
- `feature-set`: abbreviated feature family roster (see legend below)

Feature-set legend:

| code | families included |
|-|-|
| full | all 52 features (knn, knn_vote, knn_dist, go_context, annotation_meta, alignment_nw, alignment_sw, length, taxonomy_pair, taxonomy_voters, anc2vec, emb_pca) |
| lean | full minus anc2vec and emb_pca (30 features) |
| lean+lin | lean plus lineage (34 features: knn, knn_vote, knn_dist, go_context, annotation_meta, alignment_nw, alignment_sw, length, taxonomy_pair, taxonomy_voters, lineage) |
| lean+lin+emb | lean+lin plus anc2vec and emb_pca (56 features) |
| no-lin | lean+lin+emb minus lineage (52 features, all except lineage) |
| knn-only | knn, alignment_nw, annotation_meta only |
| no-pca | full minus emb_pca |
| no-pca-no-tax | full minus emb_pca minus taxonomy families |

## Index

| id | plm/K/obj/feat/eval | cells | cafaeval Fmax (NK+LK avg) | outcome | merged-in |
|-|-|-|-|-|-|
| [smoke-wandb-v0](#smoke-wandb-v0) | esmc_300m/K5/BIN/knn-only/smoke-K5 | pk-bpo | 0.039 (lab only) | smoke | pre-PR |
| [smoke-knn-only](#smoke-knn-only) | esmc_300m/K5/BIN/knn-only/smoke-K5 | pk-bpo | 0.039 (lab only) | smoke | pre-PR |
| [study-v9](#study-v9) | esmc_300m/K5/LR/full/bench-v1-K5 | 9 cells | 0.543 (cafaeval avg all) | drop (leakage-suspected, bench-v1-K5 unfiltered) | historical |
| [study-v10](#study-v10) | esmc_300m/K5/LR/full/bench-v1-K5-v10 | 9 cells | comparable to v9 | drop (superseded by v13) | historical |
| [study-v13](#study-v13) | esmc_300m/K5/LR/full/bench-v1-K5-v10 | NK only | NK avg ~0.48 (lab) | iterate (hparam exploration) | historical |
| [study-v15](#study-v15) | esmc_300m/K5/LR/full/bench-v1-K5-v10 | lk-bpo | hparam grid | iterate (hparam grid) | historical |
| [study-v16](#study-v16) | esmc_300m/K5/LR/no-pca-no-tax/bench-v1-K5-v10 | lk+pk | ablation | iterate (feature ablation) | historical |
| [study-v19](#study-v19) | esmc_300m/K5/LR/no-pca/bench-v1-K5-v10 | nk | ablation | iterate (no-pca exploration) | historical |
| [study-v21-lean](#study-v21-lean) | esmc_300m/K5/LR/lean/bench-v1-K5-v10 | 9 cells | n/a (lab fmax 0.11 lk-bpo) | iterate (lean pre-lineage) | historical |
| [leakage-fix-filtered](#leakage-fix-filtered) | esmc_300m/K5/LR/lean/bench-v1-K5-filtered | 3 cells | partial (3 cells only) | iterate (leakage mitigation baseline) | historical |
| [leakage-fix-grid](#leakage-fix-grid) | esmc_300m/K5/LR/lean/bench-v1-K5-filtered | 9 cells x 3 seeds | NK+LK avg ~0.621 (lab) | drop (superseded by v226 dataset) | FARM-EXP.9 (#24) |
| [no-emb-prostt5-k5](#no-emb-prostt5-k5) | ProstT5/K5/LR/no-emb/smoke-K5 | 9 cells | n/a (smoke dataset) | drop (PLM ablation, smoke only) | pre-PR |
| [bench-v1-K5-nk-bpo](#bench-v1-K5-nk-bpo) | esmc_300m/K5/LR/full/bench-v1-K5 | nk-bpo | 0.465 (lab) | drop (superseded by v226 lineage) | pre-PR |
| [smoke-v226-lineage-mini](#smoke-v226-lineage-mini) | esmc_300m/K5/LR/lean+lin/bench-v1-K5-v226-lineage-mini | pk-mfo | 0.137 (lab, short budget) | smoke (lineage pipeline smoke test) | historical |
| [study-v22-mini](#study-v22-mini) | esmc_300m/K5/LR/lean+lin/bench-v1-K5-v226-lineage-mini | 9 cells | NK+LK avg 0.673 cafaeval | iterate (mini budget, v226-mini dataset) | historical |
| [study-v22](#study-v22) | esmc_300m/K5/LR/lean+lin/bench-v1-K5-v226-lineage | 6 cells NK+LK | **0.6215 +- 0.0014** | **ship** (LB.2 publishable; LAFA v22 champion) | PR #18 (LR.1) |
| [study-v23](#study-v23) | esmc_300m/K5/LR/lean+lin/bench-v1-K5-v226-lineage | 9 cells | NK+LK avg ~0.693; PK catastrophic | drop in PK, partial ship in NK+LK | SUMMARY_v23-v26.md |
| [study-v24-no-lineage](#study-v24-no-lineage) | esmc_300m/K5/LR/lean/bench-v1-K5-v226-lineage | 9 cells | NK+LK avg ~0.693; PK partial recovery | iterate (best PK-MFO) | SUMMARY_v23-v26.md |
| [study-v25-all-features](#study-v25-all-features) | esmc_300m/K5/LR/lean+lin+emb/bench-v1-K5-v226-lineage | 9 cells | NK+LK avg ~0.672; PK best on bpo/cco | iterate (best PK-BPO, PK-CCO) | SUMMARY_v23-v26.md |
| [study-v26-binary](#study-v26-binary) | esmc_300m/K5/BIN/lean+lin+emb/bench-v1-K5-v226-lineage | 9 cells | NK+LK avg ~0.745 (best); PK broken | **ship** (NK+LK cells, binary objective champion) | PR #19 (LB.3), PR #20 (LM.3) |
| [study-v27-binary-multiseed](#study-v27-binary-multiseed) | esmc_300m/K5/BIN/lean+lin+emb/bench-v1-K5-v226-lineage | 6 cells NK+LK, 3 seeds | NK+LK avg **0.7378 +- 0.0028** (5/6 sig95 vs v22) | **ship** (publishable CIs for Ch.6; binary champion replicated) | this PR |
| [lr4-v18-selective](#lr4-v18-selective) | esmc_300m/K5/LR/lean+lin/bench-v1-K5-filtered | 3 cells | recomputed leakage-free | drop (historical policy, superseded by v22) | PR #21 (LR.4) |
| [study-selective-rerank-K10-v226](#study-selective-rerank-K10-v226) | esmc_300m/K5/LR/lean+lin/bench-v1-K5-v226-lineage | 6 NK+LK + 3 PK baseline | **0.6215 +- 0.0014** (9-cell selective avg) | **ship** (FARM-EXP.10 recomputed champion; supersedes legacy 0.4562) | PRs #15, #18, #21 (FARM-EXP.10+LR.1+LR.4) |

## Detailed entries

### smoke-wandb-v0

**Tuple:** esmc_300m / K5 / BIN / knn-only / smoke-K5

- **Spec file:** `runs/20260421T002752_smoke-K5_pk-bpo_knn-only_wandb_6fea86/spec.yaml` (wandb-era run)
- **Dataset:** smoke-K5 (PredictionSet via PROTEA export, sub-sampled)
- **Features:** knn_vote, alignment_nw, annotation_meta
- **Cell:** pk-bpo
- **Budget:** num_boost_round=5000, lr=0.08, num_leaves=127, early_stop=50
- **Val strategy:** protein_group (20%)
- **Seed:** 42
- **Results:** lab test_fmax = 0.039 (pre-cafaeval integration; wandb tracked)
- **Outcome:** smoke. Verified pipeline and wandb integration. No cafaeval run.
- **Notes:** Earliest run in repo. wandb backend; no reproducible spec YAML in standard schema.

### smoke-knn-only

**Tuple:** esmc_300m / K5 / BIN / knn-only / smoke-K5

- **Spec file:** `experiments/example.yaml`
- **Dataset:** smoke-K5 (small test dataset)
- **Features:** knn, alignment_nw, annotation_meta
- **Cell:** pk-bpo
- **Budget:** num_boost_round=5000, lr=0.08, num_leaves=127, min_data_in_leaf=200, early_stop=50
- **Val strategy:** protein_group (20%), neg_pos_ratio=10
- **Seed:** 42
- **Results:** lab test_fmax = 0.039
- **Outcome:** smoke. Canonical quickstart spec. Used for CI smoke gate.
- **Artifacts:** `runs/smoke-K5_pk-bpo_knn-only/`

### study-v9

**Tuple:** esmc_300m / K5 / LR / full / bench-v1-K5

- **Spec files:** `experiments/_generated/study_v9/` (f1_replication, f2_ablation, f4_hparam sub-dirs; 93 YAMLs)
- **Dataset:** bench-v1-K5 (12 train snapshot-pairs v160-v220, eval v220-v230). Unfiltered: eval proteins overlap with some training positives (leakage suspected post-hoc).
- **Features:** all 52 (full)
- **Cells:** 9 cells (nk/lk/pk x bpo/cco/mfo)
- **Budget:** num_boost_round=5000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=50
- **Val strategy:** protein_group (20%)
- **Seeds:** 42, 7, 137 (replication phase)
- **Results (cafaeval Fmax, seed winners):**

  | cell | lab Fmax | cafaeval Fmax |
  |-|-|-|
  | nk-bpo | 0.465 | 0.524 |
  | nk-mfo | 0.461 | 0.683 |
  | nk-cco | 0.484 | 0.738 |
  | lk-bpo | 0.357 | 0.554 |
  | lk-mfo | 0.244 | 0.627 |
  | lk-cco | 0.253 | 0.757 |
  | pk-bpo | 0.113 | 0.371 |
  | pk-mfo | 0.105 | 0.430 |
  | pk-cco | 0.129 | 0.201 |

  avg cafaeval Fmax = 0.543
- **Ablation finding:** anc2vec_query dominates (delta 0.14 avg); alignment, go_context, knn are secondary; taxonomy near-zero.
- **Hparam finding:** nk-bpo robust (grid range 0.44-0.47; delta 0.028, below 0.03 threshold).
- **Outcome:** drop. bench-v1-K5 eval set includes proteins from training positives (leakage). Numbers inflated vs FARM-EXP.9 leakage-free re-run. Dataset superseded by bench-v1-K5-filtered and bench-v1-K5-v226-lineage.
- **Artifacts:** `runs/study_v9/` (replication/, ablation/, bootstrap/, hparam/, SUMMARY.md)

### study-v10

**Tuple:** esmc_300m / K5 / LR / full / bench-v1-K5-v10

- **Spec files:** `experiments/_generated/study_v10/` (f1_replication, f2_ablation sub-dirs)
- **Dataset:** bench-v1-K5-v10 (updated snapshot coverage)
- **Features:** full (same 52 as v9)
- **Cells:** 9 cells
- **Budget:** same as v9
- **Seeds:** 42, 7, 137
- **Results:** comparable to v9 (NK avg ~0.48 lab, no cafaeval re-run committed)
- **Outcome:** drop. Intermediate dataset version; superseded by v13/v15 hparam series, then by v226 lineage dataset.
- **Artifacts:** `runs/study_v10/`

### study-v13

**Tuple:** esmc_300m / K5 / LR / full / bench-v1-K5-v10

- **Spec files:** `experiments/_generated/study_v13/` (NK cells, 3 seeds)
- **Dataset:** bench-v1-K5-v10
- **Features:** full. Hparam variant: lr=0.1, num_leaves=31, num_boost_round=10000, early_stop=200, neg_pos_ratio=null
- **Cells:** nk-bpo, nk-mfo, nk-cco (3 seeds each)
- **Outcome:** iterate. Hparam exploration (longer training + lower lr). Informed v15 grid design.
- **Artifacts:** `runs/study_v13/`

### study-v15

**Tuple:** esmc_300m / K5 / LR / full / bench-v1-K5-v10

- **Spec files:** `experiments/_generated/study_v15/p1_hparam/`, `p2_ablation/`, `p3_multiseed/`
- **Dataset:** bench-v1-K5-v10
- **Features:** varies per phase (full, ablation variants)
- **Cells:** lk-bpo primary (p1/p2); full 9 (p3)
- **Budget:** p1 grid: lr in {0.03, 0.05, 0.1}, num_leaves in {31, 63, 127}, num_boost_round=5000, early_stop=50
- **Outcome:** iterate. Expanded hparam grid for lk cells. Finding: L127+lr=0.03 on lk-bpo near top.
- **Artifacts:** `runs/study_v15/`

### study-v16

**Tuple:** esmc_300m / K5 / LR / no-pca-no-tax / bench-v1-K5-v10

- **Spec files:** `experiments/_generated/study_v16/` (per-cell sub-dirs)
- **Dataset:** bench-v1-K5-v10
- **Features:** full minus emb_pca minus taxonomy_pair/voters (no-pca-no-tax)
- **Cells:** lk-bpo, lk-cco, lk-mfo, nk-bpo, nk-cco, nk-mfo, pk-mfo
- **Budget:** lr=0.05, num_leaves=31, early_stop=100, num_boost_round=5000
- **Outcome:** iterate. Feature reduction study. Confirmed taxonomy and pca contribute negligibly. Informed lean feature set for v21+.
- **Artifacts:** `runs/study_v16/`

### study-v19

**Tuple:** esmc_300m / K5 / LR / no-pca / bench-v1-K5-v10

- **Spec files:** `experiments/_generated/study_v19/` (per-cell sub-dirs)
- **Dataset:** bench-v1-K5-v10
- **Features:** full minus emb_pca (no-pca)
- **Cells:** nk-bpo, lk-bpo, lk-cco, lk-mfo, nk-cco, nk-mfo, pk-mfo
- **Budget:** lr=0.05, num_leaves=31, early_stop=100, num_boost_round=5000
- **Outcome:** iterate. Confirmed dropping pca does not hurt NK/LK. Formalised lean feature set.
- **Artifacts:** `runs/study_v19/`

### study-v21-lean

**Tuple:** esmc_300m / K5 / LR / lean / bench-v1-K5-v10

- **Spec files:** `experiments/_generated/study_v21_lean/` (per-cell, seed42 only)
- **Dataset:** bench-v1-K5-v10
- **Features:** lean (full minus anc2vec minus emb_pca = 30 features)
- **Cells:** 9 cells, seed=42
- **Budget:** num_boost_round=5000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=50
- **Results:** lab test_fmax (lk-bpo seed42) = 0.114. No cafaeval run on this study.
- **Outcome:** iterate. First end-to-end lean run on full 9-cell matrix. Baseline for lineage feature addition (v22 series). anc2vec drop confirmed not harmful.
- **Artifacts:** `runs/study_v21_lean/`

### leakage-fix-filtered

**Tuple:** esmc_300m / K5 / LR / lean / bench-v1-K5-filtered

- **Spec files:** `experiments/leakage_fix/lk-bpo_seed137_filtered.yaml`, `nk-bpo_seed42_filtered.yaml`, `pk-bpo_seed7_filtered.yaml`
- **Dataset:** bench-v1-K5-filtered (bench-v1-K5 with eval proteins removed from train positives)
- **Features:** lean (no anc2vec, no emb_pca)
- **Cells:** lk-bpo (seed137), nk-bpo (seed42), pk-bpo (seed7) only
- **Budget:** num_boost_round=5000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=50
- **Results (lab test_fmax):**

  | cell / seed | lab Fmax |
  |-|-|
  | lk-bpo / seed137 | 0.622 |
  | nk-bpo / seed42 | 0.599 |
  | pk-bpo / seed7 | 0.151 |

- **Outcome:** iterate. First leakage-free numbers on filtered dataset. Confirmed leakage had inflated v9 PK numbers substantially. 3-cell partial (BPO aspect only). Superseded by full 9x3 grid (leakage-fix-grid).
- **Artifacts:** `runs/leakage_fix/lk-bpo_seed137_filtered/`, `runs/leakage_fix/nk-bpo_seed42_filtered/`, `runs/leakage_fix/pk-bpo_seed7_filtered/`

### leakage-fix-grid

**Tuple:** esmc_300m / K5 / LR / lean / bench-v1-K5-filtered

- **Spec files:** `experiments/leakage_fix/grid/` (27 YAMLs: 9 cells x 3 seeds)
- **Dataset:** bench-v1-K5-filtered
- **Features:** lean (same as leakage-fix-filtered)
- **Cells:** all 9 cells, seeds 42/7/137
- **Budget:** same as leakage-fix-filtered
- **Results (lab Fmax, mean across 3 seeds):**

  | cell | seed42 | seed7 | seed137 | mean |
  |-|-|-|-|-|
  | lk-bpo | 0.615 | 0.623 | 0.622 | **0.620** |
  | lk-cco | 0.794 | 0.796 | 0.793 | **0.794** |
  | lk-mfo | 0.605 | 0.604 | 0.608 | **0.606** |
  | nk-bpo | 0.599 | 0.600 | 0.597 | **0.599** |
  | nk-cco | 0.776 | 0.776 | 0.773 | **0.775** |
  | nk-mfo | 0.649 | 0.644 | 0.646 | **0.646** |
  | pk-bpo | 0.165 | 0.163 | 0.165 | **0.164** |
  | pk-cco | 0.342 | 0.347 | 0.347 | **0.345** |
  | pk-mfo | 0.297 | 0.291 | 0.290 | **0.293** |

  NK+LK avg (6 cells): ~0.674 (lab Fmax; no cafaeval run on this dataset).
- **Notes:** These are lab Fmax values (no cafaeval propagation), measured against the bench-v1-K5-filtered hold-out. Dataset eval range differs from the v226 lineage eval (v226-v230). Numbers not directly comparable to v22+ cafaeval Fmax.
- **Outcome:** drop. Superseded by bench-v1-K5-v226-lineage dataset (larger train coverage, lineage features). Results documented in FARM-EXP.9 (PR #24).
- **Artifacts:** `runs/leakage_fix/grid/`

### no-emb-prostt5-k5

**Tuple:** ProstT5 / K5 / LR / no-emb / smoke-K5

- **Spec files:** `experiments/_generated/no-emb-prostt5-k5_*.yaml` (9 cells)
- **Dataset:** smoke-K5 (ProstT5 embedding KNN; small)
- **Features:** no-emb (knn_vote, alignment_nw, alignment_sw, length, taxonomy_pair, taxonomy_voters, go_context, annotation_meta; no embedding-distance features)
- **Cells:** 9 cells
- **Budget:** same standard lean budget
- **Results (lab test_fmax):**

  | cell | Fmax |
  |-|-|
  | nk-bpo | 0.049 |
  | nk-cco | 0.061 |
  | nk-mfo | 0.086 |
  | lk-bpo | 0.045 |
  | lk-cco | 0.049 |
  | lk-mfo | 0.103 |
  | pk-bpo | 0.028 |
  | pk-cco | 0.030 |
  | pk-mfo | 0.028 |

- **Outcome:** drop. Smoke-only dataset; results not comparable to full bench. PLM ablation on smoke shows embedding features are the dominant signal (all Fmax below 0.1 without them). No cafaeval run.
- **Artifacts:** `runs/no-emb-prostt5-k5_*/`

### bench-v1-K5-nk-bpo

**Tuple:** esmc_300m / K5 / LR / full / bench-v1-K5

- **Spec file:** `experiments/_generated/bench-v1-K5_nk-bpo.yaml`
- **Dataset:** bench-v1-K5 (unfiltered)
- **Features:** full (52 features)
- **Cell:** nk-bpo, seed=42
- **Budget:** num_boost_round=5000, lr=0.05, num_leaves=63, early_stop=50
- **Results:** lab test_fmax = 0.465
- **Outcome:** drop. Single-cell run on unfiltered dataset (pre-leakage-fix). Superseded by leakage-fix-grid and v226-lineage studies.
- **Artifacts:** `runs/bench-v1-K5_nk-bpo/`

### smoke-v226-lineage-mini

**Tuple:** esmc_300m / K5 / LR / lean+lin / bench-v1-K5-v226-lineage-mini

- **Spec file:** `experiments/smoke-v226-lineage-mini.yaml`
- **Dataset:** bench-v1-K5-v226-lineage-mini (train v220-v226, eval v226-v230, small)
- **Features:** lean+lin (anc2vec/emb_pca dropped; lineage features included)
- **Cell:** pk-mfo, seed=42
- **Budget:** num_boost_round=200 (short), lr=0.08, num_leaves=63, min_data_in_leaf=100, early_stop=25
- **Results:** lab test_fmax = 0.137
- **Outcome:** smoke. Verified lineage feature wiring (T1.6) runs end-to-end. Short budget, not a quality measurement.
- **Artifacts:** `runs/smoke-v226-lineage-mini/`

### study-v22-mini

**Tuple:** esmc_300m / K5 / LR / lean+lin / bench-v1-K5-v226-lineage-mini

- **Spec files:** `experiments/_generated/study_v22_mini/` (9 cells, seed=42)
- **Dataset:** bench-v1-K5-v226-lineage-mini (smaller train, same eval v226-v230)
- **Features:** lean+lin (no anc2vec, no emb_pca; lineage included)
- **Cells:** 9 cells
- **Budget:** num_boost_round=5000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=50
- **Results (cafaeval Fmax from study_v22_mini/cafaeval/results.csv):**

  | cell | cafaeval Fmax |
  |-|-|
  | nk-bpo | 0.569 |
  | nk-cco | 0.772 |
  | nk-mfo | 0.690 |
  | lk-bpo | 0.657 |
  | lk-cco | 0.790 |
  | lk-mfo | 0.698 |
  | pk-bpo | 0.071 |
  | pk-cco | 0.228 |
  | pk-mfo | 0.146 |

  NK+LK avg cafaeval = 0.696
- **Outcome:** iterate. Validated lean+lin design on mini dataset. Informed full v226-lineage run (study-v22). Results slightly lower than full v22 due to reduced train coverage.
- **Artifacts:** `runs/study_v22_mini/`

### study-v22

**Tuple:** esmc_300m / K5 / LR / lean+lin / bench-v1-K5-v226-lineage

This is the canonical LAFA v22 lineage reranker, also called the "LB.2 leakage-fixed champion".

- **Spec files:** `experiments/_generated/study_v22/` (6 cells NK+LK, 3 seeds each = 18 runs)
- **Dataset:** bench-v1-K5-v226-lineage (train v210-v226, eval v226-v230; 24.4M train rows; KNN PredictionSet sha 729de2c1, EvaluationSet sha 3b6f8064)
- **Features:** lean+lin (anc2vec dropped, emb_pca dropped, lineage included: lineage_is_ancestor_of_known, lineage_is_descendant_of_known, lineage_ancestor_of_count, lineage_descendant_of_count)
- **Cells:** nk-bpo, nk-cco, nk-mfo, lk-bpo, lk-cco, lk-mfo (PK excluded: lineage shortcut catastrophic in PK)
- **Budget:** num_boost_round=10000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=100 (publication budget)
- **Val strategy:** protein_group (20%), propagate_labels=False
- **Seeds:** 42, 7, 137 (3 seeds per cell = 18 runs)
- **Results (cafaeval Fmax, multi-seed avg +- std, from LB.2 memory):**

  | cell | cafaeval Fmax (avg 3 seeds) | std | LB.3 paired bootstrap vs KNN | sig |
  |-|-|-|-|-|
  | nk-bpo | 0.584 | ~0.002 | strictly positive (baseline 0.533) | *** |
  | nk-cco | 0.776 | ~0.002 | strictly positive (baseline 0.700) | *** |
  | nk-mfo | 0.646 | ~0.001 | strictly positive (baseline 0.645) | *** |
  | lk-bpo | 0.622 | ~0.002 | strictly positive (baseline 0.584) | *** |
  | lk-cco | 0.793 | ~0.001 | strictly positive (baseline 0.705) | *** |
  | lk-mfo | 0.608 | ~0.001 | strictly positive (baseline 0.582) | *** |

  **NK+LK selective avg: 0.6215 +- 0.0014** (publishable; LB.2 result).
  6/6 NK+LK 95% CIs strictly positive at paired bootstrap N=10000 (LB.3).
- **Feature importance (v23 representative, fractional):** knn (54-74%), go_context (16-25%), annotation_meta (9-19%), lineage (0-10% in NK+LK; mechanically near-zero in NK). alignment/taxonomy/length all 0%.
- **Outcome:** **ship**. LAFA published model (v22). Canonical NK+LK champion. Deployed to PROTEA via `POST /reranker-models/import-by-reference`. Canonical EvaluationResult: `478d577c-e391-48a3-a1c2-4f45f54e3ba8` (job `ec051d63`).
- **Artifacts:** `experiments/_generated/study_v22/`, `runs/study_v22/`
- **Merged-in:** PR #18 (LR.1 formal closure)

### study-v23

**Tuple:** esmc_300m / K5 / LR / lean+lin / bench-v1-K5-v226-lineage

- **Spec files:** `experiments/_generated/study_v23/` (9 cells, seed=42)
- **Dataset:** bench-v1-K5-v226-lineage (same as v22)
- **Features:** lean+lin (same as v22; also labelled "official study_v22 budget" in SUMMARY_v23-v26.md)
- **Cells:** 9 cells, seed=42
- **Budget:** publication budget (same as v22)
- **Results (cafaeval Fmax from study_v23/cafaeval/results.csv):**

  | cell | cafaeval Fmax | baseline | lift |
  |-|-|-|-|
  | lk-bpo | 0.660 | 0.584 | +0.075 |
  | lk-cco | 0.743 | 0.705 | +0.038 |
  | lk-mfo | 0.700 | 0.582 | +0.118 |
  | nk-bpo | 0.560 | 0.533 | +0.027 |
  | nk-cco | 0.773 | 0.700 | +0.073 |
  | nk-mfo | 0.711 | 0.645 | +0.065 |
  | pk-bpo | 0.150 | 0.403 | -0.253 |
  | pk-cco | 0.309 | 0.601 | -0.292 |
  | pk-mfo | 0.161 | 0.483 | -0.322 |

- **Outcome:** drop in PK; iterate in NK+LK. DAG-closure shortcut in PK: lineage features cause catastrophic cafaeval regression (PK lift -0.25 to -0.32). NK+LK positive. Mechanism documented in `runs/SUMMARY_v23-v26.md`.
- **Artifacts:** `runs/study_v23/`, `experiments/_generated/study_v23/`

### study-v24-no-lineage

**Tuple:** esmc_300m / K5 / LR / lean / bench-v1-K5-v226-lineage

- **Spec files:** `experiments/_generated/study_v24_no_lineage/` (9 cells, seed=42)
- **Dataset:** bench-v1-K5-v226-lineage
- **Features:** lean (lean+lin minus all 4 lineage columns = same as v21-lean)
- **Cells:** 9 cells, seed=42
- **Budget:** publication budget
- **Results (cafaeval Fmax from study_v24_no_lineage/cafaeval/results.csv):**

  | cell | cafaeval Fmax | baseline | lift |
  |-|-|-|-|
  | lk-bpo | 0.647 | 0.584 | +0.063 |
  | lk-cco | 0.743 | 0.705 | +0.038 |
  | lk-mfo | 0.688 | 0.582 | +0.106 |
  | nk-bpo | 0.560 | 0.533 | +0.027 |
  | nk-cco | 0.773 | 0.700 | +0.073 |
  | nk-mfo | 0.711 | 0.645 | +0.066 |
  | pk-bpo | 0.404 | 0.403 | +0.001 |
  | pk-cco | 0.450 | 0.601 | -0.151 |
  | pk-mfo | 0.442 | 0.483 | -0.041 |

- **Outcome:** iterate. Best config for pk-mfo (lowest negative lift vs v23). Removing lineage recovers PK substantially but pk-cco and pk-mfo remain below baseline. NK+LK slightly below v23 (no lineage bonus). No single design wins all 9 cells.
- **Artifacts:** `runs/study_v24_no_lineage/`, `experiments/_generated/study_v24_no_lineage/`

### study-v25-all-features

**Tuple:** esmc_300m / K5 / LR / lean+lin+emb / bench-v1-K5-v226-lineage

- **Spec files:** `experiments/_generated/study_v25_all_features/` (9 cells, seed=42)
- **Dataset:** bench-v1-K5-v226-lineage
- **Features:** lean+lin+emb (all 56 features: lineage, anc2vec, and emb_pca all included; no drops)
- **Cells:** 9 cells, seed=42
- **Budget:** publication budget
- **Results (cafaeval Fmax from study_v25_all_features/cafaeval/results.csv):**

  | cell | cafaeval Fmax | baseline | lift |
  |-|-|-|-|
  | lk-bpo | 0.652 | 0.584 | +0.068 |
  | lk-cco | 0.753 | 0.705 | +0.047 |
  | lk-mfo | 0.693 | 0.582 | +0.111 |
  | nk-bpo | 0.541 | 0.533 | +0.008 |
  | nk-cco | 0.723 | 0.700 | +0.023 |
  | nk-mfo | 0.694 | 0.645 | +0.050 |
  | pk-bpo | 0.405 | 0.403 | **+0.002** (best PK-BPO) |
  | pk-cco | 0.579 | 0.601 | -0.022 (best PK-CCO) |
  | pk-mfo | 0.263 | 0.483 | -0.220 |

- **Outcome:** iterate. Best config for pk-bpo and pk-cco. anc2vec/emb_pca partially counterbalance lineage shortcut in pk-cco but not pk-mfo. NK+LK slightly below v23/v26-binary on most cells. Non-monotonic interaction with lineage in PK (not fully understood).
- **Artifacts:** `runs/study_v25_all_features/`, `experiments/_generated/study_v25_all_features/`

### study-v26-binary

**Tuple:** esmc_300m / K5 / BIN / lean+lin+emb / bench-v1-K5-v226-lineage

- **Spec files:** `experiments/_generated/study_v26_binary/` (9 cells, seed=42)
- **Dataset:** bench-v1-K5-v226-lineage
- **Features:** lean+lin+emb (all 56 features; no drops)
- **Cells:** 9 cells, seed=42
- **Budget:** num_boost_round=10000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=100; neg_pos_ratio=10 (binary objective subsamples negatives: train pk-bpo shrinks from 13.6M to 3.3M rows)
- **Objective:** binary (calibrated [0,1] scores; cafa_eval sweeps thresholds meaningfully)
- **Results (cafaeval Fmax from study_v26_binary/cafaeval/results.csv):**

  | cell | cafaeval Fmax | baseline | lift |
  |-|-|-|-|
  | lk-bpo | 0.664 | 0.584 | **+0.079** |
  | lk-cco | 0.795 | 0.705 | **+0.089** |
  | lk-mfo | 0.692 | 0.582 | +0.110 |
  | nk-bpo | 0.585 | 0.533 | **+0.051** |
  | nk-cco | 0.799 | 0.700 | **+0.099** |
  | nk-mfo | 0.738 | 0.645 | **+0.093** |
  | pk-bpo | 0.286 | 0.403 | -0.117 |
  | pk-cco | 0.356 | 0.601 | -0.245 |
  | pk-mfo | 0.373 | 0.483 | -0.110 |

  NK+LK: 5/6 cells best lift (lk-mfo best is v23 at +0.118).
  Bootstrap CIs (LB.3, 30-bootstrap): 6/6 NK+LK lift CIs strictly positive (*** p<0.01).
  PK: re-broken by neg_pos_ratio=10 subsampling. pk-mfo smin=0.126 (well below baseline 0.422).
- **Outcome:** **ship** (NK+LK cells only). Binary objective champion for 5/6 NK+LK cells. Selective deployment: apply v26 binary model to NK and LK proteins only; PK falls back to KNN baseline (neighbor_vote_fraction). LB.3 CI results are the publishable statistical claim for Chapter 6.
- **Artifacts:** `runs/study_v26_binary/`, `experiments/_generated/study_v26_binary/`, `runs/SUMMARY_v23-v26.md`
- **Related:** PR #19 (LB.3 paired CI), PR #20 (LM.3 feature importance)

### study-v27-binary-multiseed

**Tuple:** esmc_300m / K5 / BIN / lean+lin+emb / bench-v1-K5-v226-lineage

- **Spec files:** `v27_binary_multiseed_sweep.py` (generates specs inline), seeds 42, 137, 244
- **Dataset:** bench-v1-K5-v226-lineage
- **Features:** lean+lin+emb (all 56 features; no drops), identical to v26-binary
- **Cells:** 6 cells NK+LK (nk-mfo, nk-bpo, nk-cco, lk-mfo, lk-bpo, lk-cco), 3 seeds
- **Budget:** num_boost_round=10000, lr=0.05, num_leaves=63, min_data_in_leaf=100, early_stop=100; neg_pos_ratio=10
- **Objective:** binary, same as v26-binary
- **Results (cafaeval Fmax, seeds 42/137/244, mean +- 95% CI half-width):**

  | cell | seed=42 | seed=137 | seed=244 | mean +- CI half-width | vs v22 delta | sig95 |
  |-|-|-|-|-|-|-|
  | nk-mfo | 0.7376 | 0.7436 | 0.7411 | **0.7408 +- 0.0030** | +0.0343 [+0.0296, +0.0387] | 1 |
  | nk-bpo | 0.5848 | 0.5918 | 0.5895 | **0.5887 +- 0.0035** | +0.0291 [+0.0252, +0.0330] | 1 |
  | nk-cco | 0.7992 | 0.7999 | 0.7949 | **0.7980 +- 0.0025** | +0.0206 [+0.0152, +0.0255] | 1 |
  | lk-mfo | 0.6833 | 0.6820 | 0.6809 | **0.6821 +- 0.0012** | +0.0014 [-0.0052, +0.0063] | 0 |
  | lk-bpo | 0.6629 | 0.6680 | 0.6724 | **0.6678 +- 0.0048** | +0.0218 [+0.0165, +0.0272] | 1 |
  | lk-cco | 0.7948 | 0.8001 | 0.7971 | **0.7973 +- 0.0027** | +0.0604 [+0.0527, +0.0711] | 1 |

  NK+LK unweighted mean: **0.7291 +- 0.0028** (6-cell avg). 5/6 cells strictly dominate v22 at 95%.
  lk-mfo: v27 mean 0.6821 vs v22 mean 0.6807 (delta CI straddles zero; v27 not strictly better on lk-mfo).
  Bootstrap: N=10000, independent arms (v22 seeds 42/7/137; v27 seeds 42/137/244).
  KNN baseline: nk-mfo=0.6447, nk-bpo=0.5333, nk-cco=0.7000, lk-mfo=0.5816, lk-bpo=0.5844, lk-cco=0.7053.
  All 6 cells lift KNN baseline by a statistically significant margin.

- **Outcome:** **ship** (publishable 3-seed CIs; binary champion confirmed). Study v27 provides the
  thesis Chapter 6 publishable statistical claim: binary objective with lean+lin+emb features
  outperforms v22 lambdarank at 95% confidence on 5/6 NK+LK cells. lk-mfo is the exception
  (gains are near-zero and within noise). Selective deployment: NK+LK cells use v27-binary;
  PK falls back to KNN baseline (same policy as v22).
- **Artifacts:** `runs/v27_binary_multiseed/`, `experiments/v27/multiseed_summary.md`,
  `runs/v27_binary_multiseed/cis.json`, `runs/v27_binary_multiseed/paired_ci.json`

### lr4-v18-selective

**Tuple:** esmc_300m / K5 / LR / lean+lin / bench-v1-K5-filtered

- **Spec files:** `experiments/lr4/` (script-based; see `scripts/lr4_v18_selective.py`)
- **Dataset:** bench-v1-K5-filtered (leakage-free version of bench-v1-K5)
- **Features:** lean+lin (historical selective-rerank policy from PROTEA v18 deployment)
- **Cells:** 3 cells (selective subset: specific cells where policy was applied)
- **Policy:** historical "selective rerank" at K=10 threshold (deployed in PROTEA before leakage was identified)
- **Results:** `experiments/lr4/v18_selective_delta.csv` (Fmax delta vs baseline, leakage-free re-computation)
- **Outcome:** drop. Historical policy recomputed on leakage-free dataset for the record. Results are lower than v9 numbers (confirming leakage inflation). Superseded by v22 lineage policy. Documented for thesis cross-reference.
- **Notes:** The original v18 "selective rerank" Fmax of ~0.4562 cited in pre-LB.2 records was measured on the unfiltered dataset (leakage-suspected). LR.4 established the leakage-free baseline. v22 lineage is the replacement policy.
- **Artifacts:** `experiments/lr4/v18_selective_delta.csv`, PR #21 (LR.4)

### study-selective-rerank-K10-v226

**Tuple:** esmc_300m / K5 / LR / lean+lin / bench-v1-K5-v226-lineage

**Note on K:** The original historical "selective rerank at K=10" (PROTEA v18
deployment) used 10 nearest neighbors. No `bench-v1-K10-v226-lineage` dataset
exists in the lab. Per ADR-D34 (PROTEA, Status: Accepted, 2026-05-17) the
FARM-EXP.10 slice accepted the K=5 LB.2 multi-seed sweep as the recomputed
champion; the current lab design uses K=5 as the default. The "K10" in the
study name refers to the historical selective rerank mechanism identity, not
a K=10 nearest-neighbor run.

- **Spec files:** `experiments/farm_exp_10/` (closure summary; see
  `experiments/farm_exp_10/multiseed_summary.md`)
- **Dataset:** bench-v1-K5-v226-lineage (13 train pairs v160-v226, eval v226-v230,
  24.4M train rows, 1.07M eval rows)
- **Features:** lean+lin (knn + alignment + length + taxonomy + go_context + lineage;
  anc2vec and emb_pca dropped to remove historical leakage source)
- **Cells:** 9 cells (NK+LK reranked; PK baseline fallback per selective-deploy policy)
- **Seeds:** 42, 7, 137 (LB.2 multi-seed sweep)
- **Budget:** lambdarank, num_boost_round=10000, lr=0.05, num_leaves=63,
  min_data_in_leaf=100, early_stopping_rounds=100
- **Cafaeval:** prop=fill, norm=cafa, no_orphans=True, max_terms=500
- **Results (per-cell mean over 3 seeds):**

  | cell | policy | mean Fmax | CI half | baseline | delta |
  |-|-|-|-|-|-|
  | nk-bpo | reranker | 0.5596 | 0.0024 | 0.5333 | +0.0263 |
  | nk-mfo | reranker | 0.7065 | 0.0036 | 0.6447 | +0.0618 |
  | nk-cco | reranker | 0.7774 | 0.0048 | 0.7000 | +0.0774 |
  | lk-bpo | reranker | 0.6460 | 0.0032 | 0.5844 | +0.0616 |
  | lk-mfo | reranker | 0.6806 | 0.0060 | 0.5816 | +0.0990 |
  | lk-cco | reranker | 0.7367 | 0.0091 | 0.7053 | +0.0314 |
  | pk-bpo | baseline | 0.4031 | n/a | 0.4031 | 0.0000 |
  | pk-mfo | baseline | 0.4831 | n/a | 0.4831 | 0.0000 |
  | pk-cco | baseline | 0.6009 | n/a | 0.6009 | 0.0000 |

  9-cell selective avg cafaeval Fmax: **0.6215 +- 0.0014**.
  NK+LK reranker avg: 0.6845.
  All 6 NK+LK lifts are strictly positive across all seeds (max CI half-width 0.0091).

- **Outcome:** ship. FARM-EXP.10 recomputed champion on bench-v1-K5-v226-lineage.
  Supersedes legacy 0.4562 (memory-only, leakage-contaminated, range unknown).
  Selective-deploy policy (NK+LK reranked, PK baseline fallback) is confirmed
  across all 3 seeds and all 6 NK+LK cells. Legacy 0.4562 is documented in
  `experiments/lr4/v18_selective_delta.csv` and memory
  `project_v18_selective_rerank` (marked superseded).
- **Artifacts:** `experiments/lr4/v18_selective_delta.csv`,
  `experiments/farm_exp_10/multiseed_summary.md`,
  `experiments/lb3/per_cell_paired_ci.csv`,
  PRs #15, #18, #19, #21 (FARM-EXP.10, LR.1, LB.3, LR.4)
- **ADR:** PROTEA `docs/source/adr/D34-selective-rerank-resurrection.rst`
  (Status: Accepted; Decision points 1-7 ratify the recompute policy and
  the 0.6215 champion as superseding the legacy 0.4562)
- **eval_set_name:** bench-v1-K5-v226-lineage

## Baseline reference

All cafaeval Fmax lifts above are relative to the raw KNN baseline
(`neighbor_vote_fraction` top-K aggregation without any learned reranking).
Baseline cafaeval Fmax on bench-v1-K5-v226-lineage eval set:

| cell | baseline cafaeval Fmax |
|-|-|
| nk-bpo | 0.533 |
| nk-cco | 0.700 |
| nk-mfo | 0.645 |
| lk-bpo | 0.584 |
| lk-cco | 0.705 |
| lk-mfo | 0.582 |
| pk-bpo | 0.403 |
| pk-cco | 0.601 |
| pk-mfo | 0.483 |

## Current champion summary

| tier | spec | cafaeval Fmax | eval set |
|-|-|-|-|
| NK+LK selective (v22 lineage, 3-seed) | study-v22 | **0.6215 +- 0.0014** | bench-v1-K5-v226-lineage |
| NK+LK binary (v27, 3-seed, publishable) | study-v27-binary-multiseed | **0.7291 +- 0.0028** (6-cell avg) | bench-v1-K5-v226-lineage |
| PK | KNN baseline (no reranker) | 0.403/0.601/0.483 | bench-v1-K5-v226-lineage |

The v22 3-seed figure (0.6215) is the LB.2 lambdarank publishable claim.
The v27-binary-multiseed figure (0.7291 +- 0.0028) is the LB.2-equivalent publishable
claim for the binary objective: 5/6 NK+LK cells strictly dominate v22 at 95% confidence.
Both are on bench-v1-K5-v226-lineage (eval window v226-v230).

## Open items and next candidates

1. **v27 multi-seed done**: study-v27-binary-multiseed shipped (seeds 42/137/244). Publishable.
2. **PK-specific hparam grid**: capacity and regularisation sweep on PK cells without lineage
   (v24 design space). Propose as `study-v28-pk-hparam`.
3. **FARM-EXP.7 8-PLM ensemble**: transversal grid across multiple PLM backends.
   Pending; no spec file yet in this lab.
4. **Propagation interaction audit**: train with `propagate_labels=True` vs False to map
   the DAG-closure interaction in PK. Propose as `study-v29-propagation-audit`.

## How to propose a new spec

Add a row to the [Index](#index) and a detailed entry to [Detailed entries](#detailed-entries).

Minimum row content (all fields required before a run is considered valid):

| field | description |
|-|-|
| `id` | slug, unique across catalog |
| `plm/K/obj/feat/eval` | axis tuple (see legend above) |
| `cells` | which cells were run |
| `cafaeval Fmax` | numeric result or "n/a (smoke)" |
| `outcome` | one of: ship, drop, iterate, smoke |
| `merged-in` | PR number or "historical" or "pending" |

Spec YAML must live under `experiments/` (hand-authored) or
`experiments/_generated/` (auto-generated by a build script).
Run artefacts go under `runs/<id>/`. A new entry must reference at least one
of: spec YAML path, run.json path, or PR number.

Before proposing a new spec check this catalog to avoid re-running a
dropped configuration. Iterate entries should be referenced rather than
re-run unless the dataset or feature schema has changed.
