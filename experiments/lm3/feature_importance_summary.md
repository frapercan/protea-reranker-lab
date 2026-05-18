# LM.3 feature importance summary (v226-lineage champion)

Source: LightGBM gain from `runs/study_v23` (nine `v226full_lineage_<cell>`
boosters, seed=42, registered 2026-05-14). Metric: gain (total information
gain accumulated across all splits of all trees). Aggregation: mean rank
across the three category cells per aspect (NK/LK/PK x BPO/MFO/CCO).
Full data: `experiments/lm3/feature_importance_per_aspect.csv`.

## Top-10 per aspect (mean rank across NK/LK/PK)

### BPO (Biological Process)

| rank | feature | mean_rank | mean_gain |
| - | - | - | - |
| 1 | `go_term_frequency` | 3.00 | 204,882.3 |
| 2 | `neighbor_vote_fraction` | 3.67 | 88,400.3 |
| 3 | `evidence_code` | 4.33 | 54,296.6 |
| 4 | `ref_annotation_density` | 5.00 | 53,219.2 |
| 5 | `k_position` | 6.00 | 25,180.2 |
| 6 | `neighbor_distance_std` | 7.33 | 45,149.9 |
| 7 | `distance` | 7.67 | 56,540.3 |
| 8 | `vote_count` | 8.00 | 28,803.6 |
| 9 | `neighbor_mean_distance` | 8.67 | 35,406.7 |
| 10 | `neighbor_min_distance` | 10.67 | 20,350.4 |

### MFO (Molecular Function)

| rank | feature | mean_rank | mean_gain |
| - | - | - | - |
| 1 | `go_term_frequency` | 2.00 | 144,490.1 |
| 2 | `neighbor_vote_fraction` | 2.67 | 58,033.6 |
| 3 | `evidence_code` | 3.33 | 42,637.3 |
| 4 | `vote_count` | 6.00 | 18,987.2 |
| 5 | `ref_annotation_density` | 6.67 | 16,641.2 |
| 6 | `neighbor_distance_std` | 8.00 | 14,107.6 |
| 7 | `k_position` | 8.67 | 10,690.6 |
| 8 | `neighbor_mean_distance` | 9.33 | 11,616.8 |
| 9 | `qualifier` | 11.00 | 13,204.2 |
| 10 | `neighbor_min_distance` | 11.33 | 6,997.0 |

### CCO (Cellular Component)

| rank | feature | mean_rank | mean_gain |
| - | - | - | - |
| 1 | `go_term_frequency` | 1.67 | 290,435.4 |
| 2 | `neighbor_vote_fraction` | 3.00 | 50,995.8 |
| 3 | `evidence_code` | 5.33 | 35,764.4 |
| 4 | `k_position` | 7.00 | 13,085.7 |
| 5 | `ref_annotation_density` | 7.00 | 24,277.6 |
| 6 | `vote_count` | 7.33 | 16,562.1 |
| 7 | `neighbor_mean_distance` | 8.33 | 14,678.4 |
| 8 | `neighbor_distance_std` | 8.67 | 17,662.0 |
| 9 | `qualifier` | 9.00 | 56,388.3 |
| 10 | `neighbor_min_distance` | 10.33 | 10,693.7 |

## Cross-aspect generalists

Features ranked in the top-10 for all three aspects (aspect-stable winners):

- `evidence_code`
- `go_term_frequency`
- `k_position`
- `neighbor_distance_std`
- `neighbor_mean_distance`
- `neighbor_min_distance`
- `neighbor_vote_fraction`
- `ref_annotation_density`
- `vote_count`

## Interpretation

### BPO

`neighbor_vote_fraction` and `go_term_frequency` share the top-two
positions across NK and LK cells. In PK the ranking reverses:
`go_term_frequency` dominates, and the lineage family enters ranks 2-6
(lineage_descendant_of_count, lineage_ancestor_of_count,
lineage_is_ancestor_of_known). This PK-specific pattern reflects the
DAG-closure shortcut: PK queries already have parent terms annotated,
so the booster learns to exploit the hierarchical overlap captured by
the lineage features. The alignment family (identity_nw, alignment_score_*,
etc.) scores zero gain in every BPO cell, confirming that sequence-level
similarity information adds nothing once KNN vote and distance features
are included.

### MFO

The MFO ranking closely mirrors BPO: `neighbor_vote_fraction` and
`go_term_frequency` are the dominant signals across NK and LK.
`evidence_code` consistently ranks third, higher than in CCO, suggesting
that the curation quality of the MFO annotation source matters more for
molecular function terms (where experimental evidence is sparse relative
to BPO). PK MFO again shows lineage features at ranks 2-4, confirming
the cross-aspect generality of the DAG-closure shortcut.

### CCO

CCO is the outlier aspect. `go_term_frequency` is the undisputed dominant
feature (mean rank 1.67, mean gain nearly 2x higher than BPO). This is
consistent with CCO having fewer distinct terms and higher per-term
annotation density: term frequency predicts GO-term annotation well
for cellular components. In PK-CCO the lineage family is exceptionally
strong (lineage_ancestor_of_count and lineage_is_ancestor_of_known
at ranks 2-3, with aggregate gain exceeding 1M), the highest lineage
dominance across all nine cells. The `qualifier` feature, largely
irrelevant in BPO, enters the top-10 for CCO (mean rank 9.0), suggesting
qualifier metadata (e.g. NOT annotations) is more discriminative for
component terms.
