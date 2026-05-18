# FARM-EXP.10 selective rerank resurrection: multiseed summary

Slice: FARM-EXP.10 (farm-platform loop, phase F-EXP-RESET, priority P2).

## Scope and resolution

The original FARM-EXP.10 acceptance asked for a re-train of the
historical "selective rerank at K=10" cell (legacy champion, avg
cafaeval Fmax 0.4562, memory-only, validation range unknown,
leakage-contaminated). Per ADR-D34 (PROTEA, Status: Accepted,
2026-05-17) and the `feedback_no_archaeology_recompute` policy, the
resolution is recompute on the current leakage-fixed bench
(`bench-v1-K5-v226-lineage`) rather than reverse-engineering the
legacy K=10 axis tuple.

No K=10 dataset (`bench-v1-K10-v226-lineage`) exists in the lab. The
decision (ADR-D34, Decision point 7) accepted the K=5 LB.2 multi-seed
sweep as the FARM-EXP.10 closure: the current selective-deploy policy
(NK+LK reranked at K=5, PK baseline fallback) is the leakage-free
realisation of the historical selective rerank intent on the current
bench. Axis tuple: `plm=esmc_300m, k=5, rr=lgbm.per_cell_9.lambdarank,
feat=lean+lin (knn+alignment+taxonomy+go_context+lineage, no anc2vec/emb_pca),
eval=bench-v1-K5-v226-lineage, prop=tpr_pred, ens=none`.

## Per-cell cafaeval Fmax (LB.2 multi-seed, seeds 42 / 7 / 137)

Study: LB.2 multi-seed sweep (lambdarank, lean+lin,
bench-v1-K5-v226-lineage, 3 seeds; see spec_catalog.md `study-lb2-lambdarank`).
Eval set: bench-v1-K5-v226-lineage (eval window v226-v230).
Cafaeval: prop=fill, norm=cafa, no_orphans=True, max_terms=500.

| cell | seed=42 | seed=7 | seed=137 | mean | CI half | baseline | delta |
|-|-|-|-|-|-|-|-|
| nk-bpo | 0.5599 | 0.5571 | 0.5618 | 0.5596 | 0.0024 | 0.5333 | +0.0263 |
| nk-mfo | 0.7112 | 0.7041 | 0.7041 | 0.7065 | 0.0036 | 0.6447 | +0.0618 |
| nk-cco | 0.7733 | 0.7830 | 0.7758 | 0.7774 | 0.0048 | 0.7000 | +0.0774 |
| lk-bpo | 0.6472 | 0.6421 | 0.6485 | 0.6460 | 0.0032 | 0.5844 | +0.0616 |
| lk-mfo | 0.6877 | 0.6786 | 0.6757 | 0.6806 | 0.0060 | 0.5816 | +0.0990 |
| lk-cco | 0.7434 | 0.7252 | 0.7417 | 0.7367 | 0.0091 | 0.7053 | +0.0314 |
| pk-bpo | baseline | baseline | baseline | 0.4031 | n/a | 0.4031 | 0.0000 |
| pk-mfo | baseline | baseline | baseline | 0.4831 | n/a | 0.4831 | 0.0000 |
| pk-cco | baseline | baseline | baseline | 0.6009 | n/a | 0.6009 | 0.0000 |

NK+LK unweighted mean: 0.6845 (reranker cells only).
9-cell selective avg (NK+LK reranker + PK baseline): 0.6215 +- 0.0014.

## Comparison vs legacy champion and current champion

| champion | config | 9-cell selective avg | notes |
|-|-|-|-|
| Legacy (pre-leakage, range unknown) | K=10, leakage-contaminated | 0.4562 | superseded (ADR-D34) |
| FARM-EXP.10 recompute (LB.2, lambdarank multiseed) | K=5, leakage-fixed, v226-lineage | **0.6215 +- 0.0014** | FARM-EXP.10 live champion |
| Binary multiseed (see `experiments/v27/multiseed_summary.md`) | K=5, binary, 56 features, v226-lineage | 0.7291 +- 0.0028 (NK+LK only) | current NK+LK champion |

The FARM-EXP.10 recompute (0.6215) vs legacy leaky champion (0.4562):
delta = +0.1653. This conflates eval-distribution alignment and
leakage removal; the publishable selective-rerank lift vs the
all-baseline reference on the same bench is +0.0397.

The binary multiseed study (0.7291 NK+LK mean) supersedes on NK+LK
cells (5/6 significant). Computing 9-cell selective avg for the binary
study: adds PK baseline values (0.483, 0.403, 0.601) to the NK+LK
mean. Result: (0.7408+0.5887+0.7980+0.6821+0.6678+0.7973
+0.4831+0.4031+0.6009) / 9 = 0.6402.

## Paired CI vs LB.2 lambdarank champion

Source: `experiments/v27/multiseed_summary.md` (binary vs lambdarank comparison).
Seeds differ (binary study: 42,137,244; LB.2: 42,7,137); bootstrap is independent-arm.

| cell | binary mean | lambdarank mean | delta | 95% CI | sig_95 |
|-|-|-|-|-|-|
| nk-mfo | 0.7408 | 0.7065 | +0.0343 | [+0.0296, +0.0387] | 1 |
| nk-bpo | 0.5887 | 0.5596 | +0.0291 | [+0.0252, +0.0330] | 1 |
| nk-cco | 0.7980 | 0.7774 | +0.0206 | [+0.0152, +0.0255] | 1 |
| lk-mfo | 0.6821 | 0.6807 | +0.0014 | [-0.0052, +0.0063] | 0 |
| lk-bpo | 0.6678 | 0.6459 | +0.0218 | [+0.0165, +0.0272] | 1 |
| lk-cco | 0.7973 | 0.7368 | +0.0604 | [+0.0527, +0.0711] | 1 |

5/6 NK+LK cells show binary strictly above lambdarank at 95% confidence.
lk-mfo is the exception (delta +0.0014, CI spans zero).

## Outcome

**ship** (recomputed champion on current bench, publishable). The
LB.2 multi-seed sweep (6 NK+LK cells, seeds 42/7/137) on
`bench-v1-K5-v226-lineage` is the live FARM-EXP.10 champion.
The selective-deploy policy (NK+LK reranked, PK baseline fallback)
is confirmed by all 6 NK+LK cells showing strictly positive lift
across all seeds (max CI half-width 0.0091 on lk-cco).

The binary multiseed study supersedes on NK+LK cells (5/6 significant);
the FARM-EXP.10 recompute remains the reference for the 9-cell
selective avg narrative.

## References

- ADR-D34 (PROTEA `docs/source/adr/D34-selective-rerank-resurrection.rst`,
  Status: Accepted).
- Lab PRs: #15 (FARM-EXP.10 formalize), #18 (LR.1), #19 (LB.3),
  #21 (LR.4), #29 (binary multiseed).
- `experiments/lr4/v18_selective_delta.csv` (LR.4 canonical CSV).
- `experiments/v27/multiseed_summary.md` (binary study canonical CI table).
- `experiments/lb3/per_cell_paired_ci.csv` (LB.3 per-cell paired CI).
- Memory: `project_lb2_leakage_fixed_champion`,
  `project_v18_selective_rerank` (marked superseded),
  `project_lb3_paired_ci_2026_05_18`.
