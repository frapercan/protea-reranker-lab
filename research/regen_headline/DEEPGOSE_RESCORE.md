# DeepGO-SE + k-WTA as a pure RESCORER of the deployed pool

Does DeepGO-SE's semantic-entailment score (trained on our learned k-WTA representation + the full
554k IEA+exp corpus) convert its per-protein separability advantage into board f_micro_w when used as
a **pure rescorer** of the deployed candidate pool (one variable = the score; the candidate set is
frozen to the deployed submission, no candidate added or removed)?

Frame: TRUE board (lab obo+IA, prop=fill norm=cafa no_orphans toi, PK `-known`); cafa_eval under
`repositories/PROTEA/.venv`. Metric that decides = BP f_micro_w vs the DEPLOYED anchor. Frozen data,
read-only, CPU. Artefacts: `storage/deepgose_rescore/{rescore.py,rescore.json,control_scale.py,control_scale.json}`.

## Anchor reproduced first (precondition, exact)

| cell   | deployed anchor | A (this harness) | match |
|--------|-----------------|------------------|-------|
| LK-BPO | 0.31323         | 0.31323          | exact |
| PK-BPO | 0.14351         | 0.14351          | exact |

The A arm submits the deployed reranker file verbatim; it equals the anchor to 5 dp in both cells, so
every delta below is measured against a reproduced anchor (not the raw pool, #21).

## The rescoring result (point estimates, true frame)

| cell   | A (anchor) | B_SE (SE-only) | delta B_SE  | B_blend (rank-avg) | delta B_blend | CONTROL_random | delta ctrl |
|--------|-----------|----------------|-------------|--------------------|---------------|----------------|------------|
| LK-BPO | 0.31323   | 0.33409        | **+0.02086**| 0.32473            | +0.01150      | 0.08127        | -0.23196   |
| PK-BPO | 0.14351   | 0.11244        | **-0.03107**| 0.12781            | -0.01570      | 0.04675        | -0.09676   |

Separability that motivated the test (from `deepgose_kwta/measure_avg.json`, blind window): LK-BPO SE
AUC 0.8667 vs reranker 0.8294; PK-BPO SE 0.6263 vs reranker 0.4910 (chance).

## Paired protein-resample bootstrap CI (vs the reproduced anchor)

<!-- FILL CI -->

## Controls that make the LK positive credible (SUSPECT discipline)

- **Random-order control** (same SE score multiset, shuffled across the BP rows): LK -0.23196, PK
  -0.09676. Strongly negative in both -> the LK gain is real ORDERING, not the SE score distribution.
- **Fixed-score / scale control** (`control_scale.json`, the campaign's mandated SCALE guard):
  rank-transforming the RERANKER's own ordering to a uniform [0,1] distribution changes nothing
  (LK A_rankonly -0.00037, PK -0.00070), while SE's ordering rank-transformed keeps the full effect
  (LK B_SE_rank +0.02092 ~= raw +0.02086; PK -0.03125). The gain is SE's ordering of the in-pool
  candidates, NOT a grid/scale artefact of a different score distribution.
- **Candidate set frozen**: B_SE has the identical row set as A (LK 72,178 rows; PK 414,674), only the
  score changes. 99.85% of BP rows carry a real SE score; the <0.2% outside the 19,723-term SE vocab
  (mostly non-target proteins, metric-irrelevant) get SE=0.

## Leakage / temporal disposition (CLEAN, and symmetric)

The SE ensemble is trained on **t0 = v227** (2025-09-04) leaf-GO labels (corpus `554k_curated_v227_IEA+EXP`,
input = learned k-WTA `d8979601` codes), and the eval window is **v227 -> v230**; the new v227->v230
annotations are absent from training by construction (`deepgose_train_infer.py` header + `train_progress.json`).
This is the SAME temporal cut as the deployed reranker (`train227-test230`), so there is no differential
leakage: both scorers see t0=v227 and are judged on the blind v227->v230 delta. The blend is a FIXED
50/50 rank-average with no weight fit on anything (and no SE scores exist for past proteins to fit on),
so no eval-window tuning can sneak in. (The charter said "t0<=v225"; the realised cut is v227, the
canonical TEST frame; the principle is unchanged.)

## Independent verification

An independent adversarial verifier reimplemented the LK cell from scratch (its own OBO alt_id parser,
its own SE-lookup build, its own cafaeval driver, nothing imported from `rescore.py`) and reproduced
the result: anchor 0.313227 (matches 0.31323 to 3.1e-6), B_SE **+0.020860**, random control
**-0.235774**, with the row set asserted identical across all three arms (72,178 rows; 50,228 BP rows
with a real SE score, 1,886 forced to 0). VERDICT: VERIFIED, no harness-bug artefact.
Artefacts under `storage/deepgose_rescore/verify/`.

## Verdict

<!-- FILL VERDICT -->
