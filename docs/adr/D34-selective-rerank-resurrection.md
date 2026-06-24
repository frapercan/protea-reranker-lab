# ADR D34: Selective rerank resurrection, recompute not archaeology

**Status:** Accepted
**Date:** 2026-05-17 (proposed and accepted with LB.2 multi-seed numbers);
2026-05-18 (closure ratified by LR.4 + LB.3 paired CI);
2026-05-23 (formal run records placed under `runs/transversal/farm_exp_10/` by FARM-EXP.10)

## Context

The lab maintained a memory-only champion record for the "selective rerank at K=10"
experiment (avg cafaeval Fmax 0.4562). That record predated the range distinction
(v220/v226/v230 bands), was not generated with explicit `eval_set_name` tracking,
and was confirmed leakage-contaminated on 2026-05-05 (memory: `project_anc2vec_count_leakage`).
Any pre-fix Fmax is inflated by the anc2vec_query_known_* features, which encode
the ground-truth annotation overlap between query and training proteins.

No `bench-v1-K10-v226-lineage-prostt5` dataset exists in the lab. The legacy "K=10" label
in the champion record referred to the K-nearest-neighbours parameter of the
evaluation set, not to a distinct K=10 candidate list. The current canonical
dataset is `bench-v1-K5-v226-lineage-prostt5` (K=5 neighbours, v226 ontology snapshot,
eval window v226-v230, prostt5 embeddings).

## Decision

1. **Recompute, not archaeology.** When historical records conflict with, or
   cannot be reproduced on, current validation data, recompute on the current
   bench. Do not reverse-engineer old configurations from RerankerModel rows or
   memory entries.

2. **Selective deploy policy: NK+LK reranker, PK baseline fallback.** The
   leakage-fixed booster is deployed on the six NK and LK cells
   (no_known + limited_known). The three PK cells (previously_known) fall back
   to the KNN baseline because lineage features induce a DAG-closure shortcut
   that overfits on PK (see LM.3 feature importance audit: lineage family
   dominates PK at ranks 2-6 for all three aspects, confirming that the booster
   exploits parent-term overlap rather than generalising).

3. **Configuration:** leakage-fixed bundle on `bench-v1-K5-v226-lineage-prostt5`
   (34 features: knn, alignment, length, taxonomy, go_context, lineage;
   anc2vec_neighbor + anc2vec_query + emb_pca families dropped).
   Lambdarank LightGBM, LR=0.05, leaves=63, num_boost_round=10000,
   early_stopping_rounds=100, seeds {42, 7, 137}.

4. **Champion numbers (cafaeval Fmax, multi-seed, bench-v1-K5-v226-lineage-prostt5):**

   | cell | s42 | s7 | s137 | mean | CI_half | baseline |
   | - | - | - | - | - | - | - |
   | nk-bpo | 0.5599 | 0.5571 | 0.5618 | 0.5596 | 0.0024 | 0.5333 |
   | nk-mfo | 0.7112 | 0.7041 | 0.7041 | 0.7065 | 0.0036 | 0.6447 |
   | nk-cco | 0.7733 | 0.7830 | 0.7758 | 0.7774 | 0.0048 | 0.7000 |
   | lk-bpo | 0.6472 | 0.6421 | 0.6485 | 0.6460 | 0.0032 | 0.5844 |
   | lk-mfo | 0.6877 | 0.6786 | 0.6757 | 0.6806 | 0.0060 | 0.5816 |
   | lk-cco | 0.7434 | 0.7252 | 0.7417 | 0.7367 | 0.0091 | 0.7053 |
   | pk-bpo | baseline | baseline | baseline | 0.4031 | n/a | 0.4031 |
   | pk-mfo | baseline | baseline | baseline | 0.4831 | n/a | 0.4831 |
   | pk-cco | baseline | baseline | baseline | 0.6009 | n/a | 0.6009 |

   Aggregate:
   - 6-cell NK+LK reranker avg: **0.6845**
   - 9-cell selective avg: **0.6215 +/- 0.0014** (95% CI half-width,
     10000-iteration bootstrap of the 3-seed mean per cell)
   - 9-cell all-baseline avg: 0.5818
   - Selective lift over all-baseline: +0.0397 (publishable lift)

5. **Supersedes the legacy 0.4562 record.** The +0.1653 gross delta vs the
   legacy number conflates two effects: (a) eval distribution alignment
   (range-unknown to v226-v230), and (b) leakage removal. The publishable
   selective-rerank lift is +0.0397 over the same-bench KNN baseline.

6. **FARM-EXP.10 closure (2026-05-23).** Formal run records placed under
   `runs/transversal/farm_exp_10/` (18 runs: 6 NK+LK cells x 3 seeds).
   Summary at `experiments/farm_exp_10/summary.json`. The lab_fmax training
   metric values match the FARM-EXP.8 transversal grid (confirming reproducibility).
   The cafaeval_fmax numbers above are from the LB.2 multi-seed sweep (same
   configuration, eval via cafaeval with prop=fill, norm=cafa, no_orphans=True,
   max_terms=500, th_step=0.001).

## Consequences

**Positive:**
- Eliminates the need to reverse-engineer unknown historical configs.
- Produces a valid, reproducible champion record with full range traceability
  (`eval_set_name` pinned to `bench-v1-K5-v226-lineage-prostt5`, eval window `v226-v230`).
- Establishes the recompute-not-archaeology policy as the lab standard for
  future legacy-record conflicts.
- All 6 NK+LK cells show statistically significant lift at 95% (LB.3 paired
  bootstrap CI lower bound strictly above zero for all 6).

**Negative:**
- The legacy 0.4562 is explicitly not comparable to current champions.
- PK cells carry zero reranker delta by policy (not a null result); the
  DAG-closure shortcut is a fundamental property of the lineage feature family.

**Neutral:**
- The 9-cell selective avg (0.6215) is the primary publishable number for
  chapter 6. The 6-cell NK+LK avg (0.6845) is the secondary number showing
  the pure reranker lift without the PK baseline dilution.
- The binary multiseed study (FARM-EXP.10b/LB.2 extension) supersedes on
  NK+LK cells (5/6 significant at 95% vs lambdarank, 9-cell selective avg
  approx 0.6402). The lambdarank champion remains the reference for the
  9-cell selective avg narrative in chapter 6.

## References

- LB.2 multi-seed sweep: `docs/source/experiment_log.md` section "LB.2 multi-seed sweep"
- LR.4 closure: `docs/source/experiment_log.md` section "LR.4 closure"
- LB.3 paired CI: `experiments/lb3/per_cell_paired_ci.csv`
- LM.3 feature importance: `experiments/lm3/feature_importance_per_aspect.csv`
- FARM-EXP.10 run records: `runs/transversal/farm_exp_10/`
- FARM-EXP.10 summary: `experiments/farm_exp_10/summary.json`
- PROTEA ADR-D34: `PROTEA/docs/source/adr/D34-selective-rerank-resurrection.rst`
- Memory: `project_lb2_leakage_fixed_champion`, `feedback_no_archaeology_recompute`,
  `project_anc2vec_count_leakage`, `project_v18_selective_rerank`
