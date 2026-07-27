# Independent adversarial verification: ProtEx exemplar-verifier headline

Verifier: fresh code (`storage/protex_verify/`), read-only, frozen data, no live DB, no job
dispatch. Independent of `storage/protex/` (own kNN, postings, exemplars, model, scoring; seed 1
not 0). Compute: `repositories/PROTEA/.venv/bin/python` (cafa_eval, torch+CUDA).

## Verdict: REFUTED (anchor/frame error, not a leak)

The claim "+0.0344 (PK-BPO) / +0.0344 (LK-BPO) f_micro_w on top of the deployed reranker" is
**refuted as an additive gain over the deployed system.** The +0.034 is real ONLY against the
**raw, unfiltered reranker pool** (0.11009 / 0.28974), which ProtEx mislabelled "deployed."
Against the **canonical deployed system** (percutgraft submission, 0.14351 / 0.31323) the blend
delta is **+0.00275 (PK) / +0.00707 (LK)**, both with paired-bootstrap CI spanning zero and at or
below the cell's 0.0034 noise floor. Every leakage check passes; the bug is the anchor.

## 1. Frame reconciliation (the crux)

Independent true-frame cafa_eval (lab obo+IA, prop=fill norm=cafa no_orphans toi, PK -known):

| what | PK-BPO | LK-BPO | agrees with |
|------|--------|--------|-------------|
| raw reranker pool (`eval_scores.parquet`) = ProtEx "arm A" | **0.11009** | **0.28974** | ProtEx anchor (exact) |
| **deployed percutgraft submission** | **0.14351** | **0.31323** | calib agent (exact); percut `result_9cell` 0.14024/0.31096 (version offset) |

The canonical true-board-frame **deployed anchor is 0.14351 / 0.31323**. ProtEx's 0.110/0.290 is
the raw pool, sitting 0.033 (PK) / 0.023 (LK) below the real deployed system, because the deployed
submission applies a prefilter + percutgraft the raw pool never received. The "three anchors":
0.110/0.290 = raw pool; 0.117/0.348 = a *stale* `board` field stored inside `result_9cell.json`
(superseded); 0.140/0.313 = the real deployed reconstruction (three harnesses agree).

## 2. Independent deltas vs BOTH anchors

**PK-BPO** (n=616,223) A_raw=0.11009, D_deployed=0.14351:

| arm | f | delta vs A_raw | delta vs D_deployed |
|-----|---|----------------|---------------------|
| with_neg | 0.11739 | +0.0073 [0.001, 0.014] | -0.0261 [-0.034, -0.018] |
| no_neg | 0.10794 | -0.0022 | -0.0356 |
| **blend** | 0.14626 | **+0.0362 [0.032, 0.040]** | **+0.0028 [-0.004, +0.010]** |
| random | 0.03413 | -0.0760 | -0.1094 |

**LK-BPO** (n=52,011) A_raw=0.28974, D_deployed=0.31323:

| arm | f | delta vs A_raw | delta vs D_deployed |
|-----|---|----------------|---------------------|
| with_neg | 0.31648 | +0.0267 [0.003, 0.050] | +0.0033 [-0.016, +0.024] |
| no_neg | 0.28421 | -0.0055 | -0.0290 |
| **blend** | 0.32029 | **+0.0306 [0.016, 0.047]** | **+0.0071 [-0.006, +0.021]** |
| random | 0.07766 | -0.2121 | -0.2356 |

**Reproduction:** blend-vs-raw reproduces ProtEx (mine +0.036/+0.031 vs their +0.034/+0.034) ->
ProtEx's internal computation has no leak/bug. But **blend-vs-deployed is +0.003/+0.007, both CI
spanning zero** -> the gain over the actual deployed system is null. The isolated with_neg PK lever
(ProtEx +0.0226) does NOT reproduce (mine +0.0073) and is measured off the wrong anchor regardless.

## 3. Leak / artifact audit (all CLEAN)

- Self-as-own-exemplar: `self_in_neighbours=0`, `selfex_leak=0` across train/val/test. CLEAN.
- Near-dup filter: max retained neighbour cosine 0.99951 (cos>=0.9999 dropped). CLEAN.
- Negatives label-derived? No. Postings built strictly from v227 (t0) `reference_annotations`
  propagated through the obo; negatives = t0 neighbours lacking the term; blind v227-v230 labels
  never touched during exemplar construction (independently re-derived). CLEAN.
- Temporal gate: train v<=225, early-stop v225-v227, blind v227-v230. CLEAN.
- Random-order control strongly negative (-0.076 / -0.212) -> the calibrate-onto-histogram trick
  cannot manufacture a gain. CLEAN.

The refutation is **not a data leak** - it is a frame/anchor error.

## 4. Mechanism: the verifier substitutes for the prefilter

Blend vs raw pool: newly-crosses candidates 3.5x more often true than the ones it drops (PK
10.9% vs 3.1%; LK 27.2% vs 8.3%). This is a genuine precision-improving reorder, not an artifact.
But it only lifts the raw pool UP TO the level the deployed prefilter+percutgraft already reaches
(0.146 ~ 0.14351; 0.320 ~ 0.31323). The "tiny AUC delta -> large f jump" resolves as: the f jump
0.110->0.146 recovers the prefilter, which deployed already has. The verifier is a substitute for
the deployed prefilter, not an addition on top of it.

## 5. Board implication vs TransFew (honest)

Board deployed PK-BPO 0.2181 / LK-BPO 0.4402; TransFew 0.2943 / 0.5120 (gap -0.0762 / -0.0718).
Verified true-frame gain over deployed = +0.003 / +0.007, both CI-spanning-zero. Even a fictional
1:1 transfer moves PK-BPO 0.2181->~0.221 and LK-BPO 0.4402->~0.447: **neither closes nor flips the
gap to TransFew, at most trivially narrows** (still ~-0.073 / -0.065 behind), and realistically
transfers to ~0 (consistent with the standing finding that cell-frame lab deltas do not reach the
board).

## Receipts
- Code: `storage/protex_verify/{score_anchor,build_verify,eval_verify}.py`
- Results: `storage/protex_verify/{anchor_reconciliation,eval_result_indep,build_meta}.json`
- Canonical deployed: `repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/`

Process note: `calibrate()` (shared by ProtEx and this harness) calls `np.quantile(ref, q)` with a
~600k-element `q` = O(n^2), which hangs; replaced here with exact O(n log n) monotone rank->order
mapping (corr 0.99999, identical ordering). ProtEx's stored numbers are unaffected (its run
completed, just slowly).
