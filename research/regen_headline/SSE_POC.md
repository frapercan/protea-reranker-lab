# SSE (Sparse Semantic-Entailment): a from-scratch POC on a subsample

Does an ultra-light model that fuses the two-tower's SPARSE/interpretable representation with
DeepGO-SE's logical entailment, expressed as SPARSE SET CONTAINMENT, separate and convert on the BP
cells, before spending an overnight full-scale run?

**One line.** SSE **TRAINS**, **SEPARATES** as well as DeepGO-SE (PK-BPO per-protein AUC 0.613 vs
DeepGO-SE 0.626, beating the deployed reranker's 0.478 by +0.135) and stays **INTERPRETABLE** (co-process
BP terms share ~2x more active sparse dims than unrelated pairs), but does **NOT CONVERT**: every
true-frame f_micro_w delta vs the deployed anchor is negative (rescore PK -0.0253, CI [-0.0322,-0.0183],
p>0=0; generator negative in both combines at every K). The one positive (+0.0028) is a fast-engine
artifact that contradicts the deciding cafa_eval and is flagged as a suspect. **Verdict: a full-scale run
is NOT justified for the headline; this is a second, architecturally-INDEPENDENT confirmation that the BP
wall is a separability/precision limit at the reachable tail, not a machinery/representation deficiency.**

## The model (built exactly to the SSE spec, distinct from the prior ball-space ENTAIL_KWTA)

Everything is a sparse SDR (width N=2048). Entailment = sparse set containment; axioms = differentiable
soft-containment penalties; ensemble consensus = MIN over K seeds.

- **Protein tower** (from scratch): dense ESM2-3B mean, 2560-d (`generator_frames/raw_esm2_3b.npy`) ->
  MLP (2560->1024->2048) -> soft k-WTA(top-128) -> sparse protein code x_p.
- **Term tower** (learned end-to-end): embedding table (n_terms x 2048) -> soft k-WTA with a **per-term
  bit budget m_t in [16,112] set by ontology depth** (general terms few bits, specific terms many bits) ->
  sparse term code z_t. This per-term cardinality is REQUIRED: with equal m the subsumption
  supp(z_D) subset supp(z_C) forces identical codes (degenerate).
- **Axiom regularizer** (t0 2025-07-22 GO EL normal forms `go_2025-07-22.norm`): nf1 subsumption
  supp(z_D) subset supp(z_C); nf4 existential supp(R_r . z_D) subset supp(z_C) with a learned sparse
  relation mask R_r. Enforced as `relu(g_D - g_C)` with the **parent gate detached** (containment is the
  child's duty: the child must cover the parent's features; without the detach the penalty is minimized
  degenerately by pushing the parent DOWN and hard containment gets WORSE).
- **Entailment scorer**: s(p,t) = |supp(x_p) cap supp(z_t)| / |supp(z_t)| (soft, differentiable for
  training; ensemble MIN over 5 seeds at inference).
- **Loss** = weighted-BCE(coverage) + margin(true > hardest negatives) + axiom-containment penalty.

## Subsample (fast POC, ~4 min end-to-end on one 12 GB GPU)

- **25,000 training proteins**, a subset of the 88k experimental `generator_frames` set, with the
  **4,925 LK+PK eval proteins HELD OUT of training** (no per-protein memorization).
- **3,179 BP vocab terms** (>=25 positives within the subsample). Axioms in-vocab: nf1 5,980, nf4 1,398.
- Eval: all **523 LK-BPO** + **4,402 PK-BPO** targets scored; separability on the reachable tail
  (513 LK / 2,608 PK proteins with >=1 true and >=1 false pool BP term).
- 5 independently-seeded code sets x 12 epochs.

## (a) TRAINS + axioms satisfied? YES

- Mean val AUC **0.9206** across 5 seeds (0.9181-0.9249); loss decreases monotonically.
- Axiom containment (hard codes, seed 0) vs a shuffled-pair random floor: **nf1 subsumption 0.407 vs
  0.113 (3.6x)**, **nf4 existential 0.806 vs 0.014**. The penalty raises containment far above chance;
  exact full-subset is rare for nf1 (independent per-term embeddings, near-equal cardinalities), and the
  soft penalty it directly optimizes is the enforced quantity.

## (b) SEPARABILITY (diagnostic) — reaches DeepGO-SE, beats the reranker

| cell | SSE AUC (min) | SSE AUC (avg) | reranker | SSE-reranker | vs DeepGO-SE 0.626 |
|---|---|---|---|---|---|
| LK-BPO (n=513)  | 0.8317 | 0.8435 | 0.8134 | +0.018 / +0.030 | — |
| PK-BPO (n=2608) | **0.6129** | 0.6171 | 0.4779 | **+0.135** | -0.013 / -0.009 |

PK-BPO clean sparse-containment entailment reaches **0.613**, matching DeepGO-SE's 0.626 and far above the
deployed reranker's 0.478 (chance). Because SSE shares NO machinery with the prior ball-space model, this
is an architecturally independent reproduction of the entailment-separability signal.

## (c) CONVERSION (DECIDES) — NO, every true-frame delta is negative

Frame: prop=fill norm=cafa no_orphans toi, PK exclude=groundtruth_PK_known, `cafa_eval` under
PROTEA/.venv, temporal gate v227->v230. Anchor reproduced EXACTLY: **LK 0.31323 / PK 0.14351**.

**Rescore** (rank-match the reranker's score multiset in SSE order; isolates ranking, adds no terms):

| cell | anchor | SSE-order delta [CI] p>0 | random-order control |
|---|---|---|---|
| PK-BPO | 0.14351 | **-0.0253** [-0.0322, -0.0183] p>0=0.0 | -0.0571 |
| LK-BPO | 0.31323 | -0.0416 (no -known; suspect memorization, still negative) | — |

SSE order beats random-order (-0.025 vs -0.057) -> the 0.613 separability is real -> but it still loses
to the deployed reranker at the metric's operating point.

**Generator** (union top-k SSE proposals into the pool; `cafa_eval`, authoritative):

| cell | top5 (min/avg) | top10 | top25 | random-order (top5) | matched-volume (top5) |
|---|---|---|---|---|---|
| PK-BPO | -0.0081 / -0.0035 | -0.021 / -0.014 | -0.047 / -0.040 | -0.008 / -0.004 | -0.071 |
| LK-BPO | -0.0027 / -0.0017 | -0.0043 | -0.0034 | -0.006 | -0.149 |

Negative in both combines at every K, barely above the random-order control (the SSE ordering adds almost
nothing the metric rewards), catastrophic under matched-volume.

**Flagged suspect (discipline).** The fast `measure_pk` decomposition reported the PK generator top5 at
**+0.00284** (CI [0.0017,0.0041]). It is DISCARDED: `measure_pk` omits cafa_eval's cafa-coverage
normalization penalty on ADDED predictions, so it disagrees in SIGN with the deciding cafa_eval engine
(-0.0081 min / -0.0035 avg) for the identical extras. A positive that contradicts the true frame is a
suspect, not a conversion. The rescore CI is retained because rescoring adds no rows, so the two engines
do not diverge there (and the rescore is negative anyway).

## (d) INTERPRETABILITY (bonus) — YES, sparse dims are functional modules

Hard term-code Jaccard (seed 0), co-process vs unrelated, random-pair floor **0.038**:

| pair | jaccard |
|---|---|
| glycolysis ~ gluconeogenesis (carb metab) | 0.068 |
| glycolysis ~ TCA cycle (energy) | 0.088 |
| translation ~ rRNA processing (ribosome) | 0.088 |
| glycolysis ~ translation (unrelated) | 0.029 |
| translation ~ fatty-acid beta-ox (unrelated) | 0.036 |

Co-process BP terms share ~2x more active dims than unrelated pairs, which sit at the random floor. The
sparse dimensions behave like shared functional modules.

## VERDICT

SSE **separates** (PK 0.613 ~ DeepGO-SE 0.626, beats reranker +0.135), **trains** with satisfied axioms,
and stays **interpretable** — but does **NOT convert** (all true-frame f_micro_w deltas vs the deployed
anchor negative; the one positive is a fast-engine artifact). **A full-scale run is NOT justified for the
headline f_micro_w.** This is a second, architecturally-independent confirmation that the BP wall is a
separability/precision limit at the reachable tail, not a machinery or representation deficiency
(corroborating the prior ball-space ENTAIL_KWTA "separates-but-capped" result with an entirely different
mechanism). The separability + interpretability are thesis-worthy as a mechanism/diagnostic contribution
and should be independently verified before any claim.

## Receipts

- Model + training + inference: `storage/sse_poc/sse_train_infer.py`; diagnostics
  `storage/sse_poc/sse_diagnostics.json`; scores `sse_scores_{LK,PK}.npz`; checkpoints `models/sse_*.th`.
- Separability + generator (cafa_eval): `storage/sse_poc/sse_measure.py`, `measure_{min,avg}.json`.
- Rescore + bootstrap CI: `storage/sse_poc/sse_convert_ci.py`, `convert_ci.json`.
- Consolidated: `storage/regen_headline/SSE_POC.json` (and `storage/sse_poc/SSE_POC.json`).
- Frozen data only (dense ESM2-3B, t0/v227 labels, GO EL norms, lafa_gt, deployed predictions); no live
  DB, no job dispatch. Compute: `repositories/PROTEA/.venv`.
