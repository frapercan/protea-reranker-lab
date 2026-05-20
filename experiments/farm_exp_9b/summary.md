# FARM-EXP.9b harvest summary

Generated: 2026-05-18. Runner: farm-exp-9b-runner (tmux). Total cells in scope: 94.
Final runner state: 94/94 ok.

## Caveat: non-comparability

bench-v1-K5 and bench-v1-K5-filtered numbers are NOT comparable (thesis table `tab:improvement`). The old pre-leakage Fmax used bench-v1-K5 (52 features, unfiltered, leakage-present) evaluated with prop=tpr_pred. The new leakage-fixed Fmax uses bench-v1-K5-filtered (30 features lean, v6+anc2vec-leakfree, filtered to remove leakage) with prop=tpr_pred. The paired CI below compares the SAME cell config, leakage-fixed vs not, on bench-v1-K5-filtered. The large positive deltas reflect removal of inflated pre-leakage scores and are expected.

## Replication cells (27 runs: 9 cells x 3 seeds)

All cells: bench-v1-K5-filtered, v6+anc2vec-leakfree features (30 features lean), lambdarank, L=63, lr=0.05.
Paired CI: leakage-fixed (new) vs pre-leakage (old, bench-v1-K5). NOT comparable eval sets.

| cell | seed | new Fmax | prior Fmax (bench-v1-K5, NOT comparable) |
| - | - | - | - |
| nk-bpo | 42 | 0.5890 | 0.4649 |
| nk-bpo | 7 | 0.5849 | 0.4638 |
| nk-bpo | 137 | 0.5881 | 0.4636 |
| nk-mfo | 42 | 0.6346 | 0.4592 |
| nk-mfo | 7 | 0.6321 | 0.4605 |
| nk-mfo | 137 | 0.6375 | 0.4570 |
| nk-cco | 42 | 0.7741 | 0.4835 |
| nk-cco | 7 | 0.7735 | 0.4818 |
| nk-cco | 137 | 0.7749 | 0.4840 |
| lk-bpo | 42 | 0.6157 | 0.3505 |
| lk-bpo | 7 | 0.6139 | 0.3494 |
| lk-bpo | 137 | 0.6157 | 0.3570 |
| lk-mfo | 42 | 0.5729 | 0.2440 |
| lk-mfo | 7 | 0.5741 | 0.2411 |
| lk-mfo | 137 | 0.5740 | 0.2395 |
| lk-cco | 42 | 0.7891 | 0.2491 |
| lk-cco | 7 | 0.7874 | 0.2503 |
| lk-cco | 137 | 0.7879 | 0.2525 |
| pk-bpo | 42 | 0.1314 | 0.1110 |
| pk-bpo | 7 | 0.1314 | 0.1125 |
| pk-bpo | 137 | 0.1368 | 0.1129 |
| pk-mfo | 42 | 0.2853 | 0.1047 |
| pk-mfo | 7 | 0.2790 | 0.1045 |
| pk-mfo | 137 | 0.2814 | 0.1034 |
| pk-cco | 42 | 0.2787 | 0.1248 |
| pk-cco | 7 | 0.2788 | 0.1238 |
| pk-cco | 137 | 0.2855 | 0.1285 |

### Per-cell aggregated Fmax (leakage-fixed, bench-v1-K5-filtered)

| cell | tier | aspect | n seeds | new Fmax mean | 95% CI half | prior mean (NOT comparable) | paired delta mean | paired CI lo | paired CI hi | sig 95 |
| - | - | - | - | - | - | - | - | - | - | - |
| nk-bpo | NK | BPO | 3 | 0.5874 | 0.0021 | 0.4641 | 0.1232 | 0.1211 | 0.1245 | 1 |
| nk-mfo | NK | MFO | 3 | 0.6347 | 0.0027 | 0.4589 | 0.1758 | 0.1715 | 0.1805 | 1 |
| nk-cco | NK | CCO | 3 | 0.7742 | 0.0007 | 0.4831 | 0.2911 | 0.2906 | 0.2917 | 1 |
| lk-bpo | LK | BPO | 3 | 0.6151 | 0.0009 | 0.3523 | 0.2628 | 0.2588 | 0.2652 | 1 |
| lk-mfo | LK | MFO | 3 | 0.5737 | 0.0006 | 0.2416 | 0.3321 | 0.3289 | 0.3345 | 1 |
| lk-cco | LK | CCO | 3 | 0.7881 | 0.0009 | 0.2506 | 0.5375 | 0.5354 | 0.5400 | 1 |
| pk-bpo | PK | BPO | 3 | 0.1332 | 0.0027 | 0.1121 | 0.0211 | 0.0189 | 0.0239 | 1 |
| pk-mfo | PK | MFO | 3 | 0.2819 | 0.0032 | 0.1042 | 0.1777 | 0.1745 | 0.1807 | 1 |
| pk-cco | PK | CCO | 3 | 0.2810 | 0.0034 | 0.1257 | 0.1553 | 0.1538 | 0.1570 | 1 |

### Aspect-aggregated summary

| tier | n seeds | new Fmax mean (leakage-fixed) | prior delta mean (NOT comparable) |
| - | - | - | - |
| NK | 9 | 0.6654 | 0.1967 |
| LK | 9 | 0.6590 | 0.3775 |
| PK | 9 | 0.2320 | 0.1180 |

## Ablation study (39 runs: 3 cells x 13 feature families)

One feature family dropped per run vs full model (30 features). Delta = ablated Fmax minus full model mean Fmax.
Ablation cells: lk-cco (LK representative), nk-bpo (NK representative), pk-mfo (PK representative).

| cell | family dropped | Fmax | full model Fmax | delta |
| - | - | - | - | - |
| lk-cco | alignment_nw | 0.7902 | 0.7881 | 0.0021 |
| lk-cco | alignment_sw | 0.7895 | 0.7881 | 0.0014 |
| lk-cco | anc2vec_neighbor | 0.7891 | 0.7881 | 0.0010 |
| lk-cco | anc2vec_query | 0.7891 | 0.7881 | 0.0010 |
| lk-cco | annotation_meta | 0.7932 | 0.7881 | 0.0051 |
| lk-cco | emb_pca | 0.7891 | 0.7881 | 0.0010 |
| lk-cco | go_context | 0.7788 | 0.7881 | -0.0093 |
| lk-cco | knn | 0.7612 | 0.7881 | -0.0269 |
| lk-cco | knn_distance | 0.7857 | 0.7881 | -0.0024 |
| lk-cco | knn_vote | 0.7652 | 0.7881 | -0.0229 |
| lk-cco | length | 0.7882 | 0.7881 | 0.0001 |
| lk-cco | taxonomy_pair | 0.7877 | 0.7881 | -0.0004 |
| lk-cco | taxonomy_voters | 0.7876 | 0.7881 | -0.0005 |
| nk-bpo | alignment_nw | 0.5878 | 0.5874 | 0.0005 |
| nk-bpo | alignment_sw | 0.5922 | 0.5874 | 0.0048 |
| nk-bpo | anc2vec_neighbor | 0.5890 | 0.5874 | 0.0017 |
| nk-bpo | anc2vec_query | 0.5890 | 0.5874 | 0.0017 |
| nk-bpo | annotation_meta | 0.5850 | 0.5874 | -0.0023 |
| nk-bpo | emb_pca | 0.5890 | 0.5874 | 0.0017 |
| nk-bpo | go_context | 0.5869 | 0.5874 | -0.0005 |
| nk-bpo | knn | 0.5874 | 0.5874 | 0.0000 |
| nk-bpo | knn_distance | 0.5841 | 0.5874 | -0.0032 |
| nk-bpo | knn_vote | 0.5876 | 0.5874 | 0.0002 |
| nk-bpo | length | 0.5880 | 0.5874 | 0.0006 |
| nk-bpo | taxonomy_pair | 0.5924 | 0.5874 | 0.0050 |
| nk-bpo | taxonomy_voters | 0.5875 | 0.5874 | 0.0001 |
| pk-mfo | alignment_nw | 0.2897 | 0.2819 | 0.0078 |
| pk-mfo | alignment_sw | 0.2893 | 0.2819 | 0.0074 |
| pk-mfo | anc2vec_neighbor | 0.2853 | 0.2819 | 0.0034 |
| pk-mfo | anc2vec_query | 0.2853 | 0.2819 | 0.0034 |
| pk-mfo | annotation_meta | 0.2763 | 0.2819 | -0.0056 |
| pk-mfo | emb_pca | 0.2853 | 0.2819 | 0.0034 |
| pk-mfo | go_context | 0.2609 | 0.2819 | -0.0210 |
| pk-mfo | knn | 0.2431 | 0.2819 | -0.0388 |
| pk-mfo | knn_distance | 0.2817 | 0.2819 | -0.0002 |
| pk-mfo | knn_vote | 0.2434 | 0.2819 | -0.0385 |
| pk-mfo | length | 0.2811 | 0.2819 | -0.0008 |
| pk-mfo | taxonomy_pair | 0.2822 | 0.2819 | 0.0003 |
| pk-mfo | taxonomy_voters | 0.2775 | 0.2819 | -0.0044 |

### Most impactful feature families (ablation delta)

**lk-cco** (full model mean Fmax: 0.7881):
  - Most critical (worst drop when removed): knn (delta=-0.0269)
  - Least critical (neutral or positive): annotation_meta (delta=0.0051)

**nk-bpo** (full model mean Fmax: 0.5874):
  - Most critical (worst drop when removed): knn_distance (delta=-0.0032)
  - Least critical (neutral or positive): taxonomy_pair (delta=0.0050)

**pk-mfo** (full model mean Fmax: 0.2819):
  - Most critical (worst drop when removed): knn (delta=-0.0388)
  - Least critical (neutral or positive): alignment_nw (delta=0.0078)

## Hparam sweep (27 runs: 3 num_leaves x 3 lr x 3 neg_pos_ratio, nk-bpo seed42)

Baseline: nk-bpo seed42 default hparams (L=63, lr=0.05, npr=None), Fmax=0.5890.

| leaves | lr | neg_pos_ratio | Fmax | delta vs baseline |
| - | - | - | - | - |
| 31 | 0.1 | 5 | 0.5946 | 0.0056 |
| 31 | 0.1 | none | 0.5946 | 0.0056 |
| 31 | 0.1 | 10 | 0.5946 | 0.0056 |
| 63 | 0.1 | none | 0.5922 | 0.0031 |
| 63 | 0.1 | 5 | 0.5922 | 0.0031 |
| 63 | 0.1 | 10 | 0.5922 | 0.0031 |
| 127 | 0.1 | 10 | 0.5921 | 0.0030 |
| 127 | 0.1 | 5 | 0.5921 | 0.0030 |
| 127 | 0.1 | none | 0.5921 | 0.0030 |
| 127 | 0.05 | none | 0.5913 | 0.0023 |
| 127 | 0.05 | 10 | 0.5913 | 0.0023 |
| 127 | 0.05 | 5 | 0.5913 | 0.0023 |
| 63 | 0.05 | 10 | 0.5890 | 0.0000 |
| 63 | 0.05 | none | 0.5890 | 0.0000 |
| 63 | 0.05 | 5 | 0.5890 | 0.0000 |
| 31 | 0.05 | 10 | 0.5886 | -0.0005 |
| 31 | 0.05 | none | 0.5886 | -0.0005 |
| 31 | 0.05 | 5 | 0.5886 | -0.0005 |
| 127 | 0.003 | 5 | 0.5871 | -0.0019 |
| 127 | 0.003 | 10 | 0.5871 | -0.0019 |
| 127 | 0.003 | none | 0.5871 | -0.0019 |
| 63 | 0.003 | 10 | 0.5860 | -0.0031 |
| 63 | 0.003 | none | 0.5860 | -0.0031 |
| 63 | 0.003 | 5 | 0.5860 | -0.0031 |
| 31 | 0.003 | none | 0.5812 | -0.0078 |
| 31 | 0.003 | 10 | 0.5812 | -0.0078 |
| 31 | 0.003 | 5 | 0.5812 | -0.0078 |

Best hparam config: L=31, lr=0.1, npr=5, Fmax=0.5946 (delta=0.0056 vs baseline L=63 lr=0.05 npr=None).

## Standalone run

- farm_exp_9_standalone_nk-bpo_seed42: Fmax=0.5890

## Champion table diff

The FARM-EXP.4 champion auto-promoter (`scripts/update_champions.py`) requires FARM-EXP.3 format run records (with `fmax_samples` array and canonical `axis` block). The FARM-EXP.9b run.json files use the simpler transversal format (single `metrics.test_fmax` scalar). No FARM-EXP.3 records were produced in this pass.

**Champion table rows changed: 0** (auto-promoter requires FARM-EXP.5 writer slice to produce FARM-EXP.3 format).

The existing `champions.md` LM.1 bootstrapped table (study_v23, bench-v1-K5-v226-lineage-prostt5) remains the current champion record. FARM-EXP.9b results use bench-v1-K5-filtered and are not directly comparable.
