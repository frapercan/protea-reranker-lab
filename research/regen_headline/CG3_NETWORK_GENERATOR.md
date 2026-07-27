# CG.3 -- Network channel generator: VERDICT

**GO/NO-GO: NO-GO.** The biologically-correct network channel (a protein's already-annotated
PPI / co-expression partners) is the **strongest of the five generation channels tested**, clearing
the co-occurrence floor by 2.3-3.6x with provably leakage-free signal, yet it **still cannot separate
the BP tail**: 94-98% of the proposed IA-mass is false, precision stays far below the 11.59% classifier
that itself did not convert, and unioning the proposals **lowers f_micro_w in every arm**. The BP
separability wall extends to the network axis, which was the last untested orthogonal channel.

## The one question
Does a pre-t0 PPI/co-expression NETWORK generate the true BP-tail terms our sequence-only pool misses,
at a precision high enough to clear the fill-tax bar, where four prior channels failed? **No.**

## Download actually obtained
STRING v12.0 per-species `protein.links.detailed` + `protein.aliases` (release 2023-07-26, provably
pre-t0 = 2025-09-04, CC BY 4.0). **18 tier-1 taxa, 1.7 GB** under `storage/string_v12/`. Taxon-id
remaps applied at download: `559292 -> 4932` (S. cerevisiae S288C), `83333 -> 511145` (E. coli K-12
MG1655); S. pombe kept at `284812`. The 18 taxa cover **97.6%** of the 4,925 LK+PK-BPO targets' taxa.
Realised **proposal coverage** (targets with >=1 clean-channel proposal outside the pool) is
**76.7% (LK) / 82.6% (PK)** -- lower than taxon coverage because a proposal also needs partners
carrying frozen BP annotations for terms not already in the pool. The ~2% tail taxa (mostly singleton
non-model organisms) were not downloaded; they contribute 0 proposals and cannot change the verdict.

## Generator
For target `p` with STRING partners `q`, each partner contributes its FROZEN t0 BP annotations `B(q)`:
`s(t|p) = sum_q w(p,q) * 1[t in B(q)]`, `w` = noisy-OR of the retained channel probabilities, edges
kept at `w >= 0.40`. **HEADLINE arm = experimental + coexpression channels ONLY** (provenance-clean).
`textmining` is never read. `database` appears only in a flagged ablation. Metric machinery (obo BP
ancestors, IA, gt/pool propagation, added-true vs added-false IA-mass) is reused **verbatim** from
`cg1_step1_gate.py`; only the proposal source changed (partners' `B(q)` instead of PPMI seed->BP).

## Step 1 -- added-true IA-precision (the gate)

HEADLINE clean arm (exp+coexp), added-true / (added-true + added-false):

| cell | top5 | top10 | top25 | top50 | co-occ bar (CG.1) | lit bar (CG.2) | classifier bar |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LK-BPO | **0.0578** | 0.0551 | 0.0342 | 0.0210 | 0.0161 | 0.0188 | 0.1159 |
| PK-BPO | **0.0241** | 0.0243 | 0.0172 | 0.0138 | 0.0103 | 0.0149 | 0.1159 |

The clean channel **clears the ~1% co-occurrence floor clearly (the first channel to do so)** but does
**not approach** the 11.59% classifier bar. FN IA-mass recovered at top50 is only 12.4% (LK) / 6.5% (PK)
of the pool's misses. Even where precision is highest the false tail is 13-40x the true tail.

**Database ablation (clean + database).** top5 = 0.0776 (LK) / 0.0326 (PK): only modestly above the
clean arm. The clean exp+coexp channels alone carry the signal, so the result is **not** database
circularity (pathway/GO-derived edges).

**Stratification.** Signal concentrates in densely-curated taxa (human LK top5 13.8%, rat 17.9%) and
mid hub-degree bands (PK 51-200 partners = 9.6%); sparse taxa contribute ~0 (Arabidopsis 0%, most tail
taxa 0%). Consistent with "process is a network property where the interactome is well-annotated," but
it does not separate at the tail.

## Step 2 -- does it convert? (true board frame)
Union the clean network proposals into the deployed pool; **one variable = candidate set only**; extras
scored by their own min-max-normalised network confidence (DB-free analog of the template's raw-sigmoid
arm; no live DB, no transfer scorer). cafa_eval under `repositories/PROTEA/.venv/bin/python`, lab
obo+IA, `prop=fill norm=cafa no_orphans toi`; **PK adds `-known`** (`evaluation.nf:279`).

| cell | A (pool only) | top5 delta | top10 delta | top25 delta |
|------|:---:|:---:|:---:|:---:|
| LK-BPO | 0.31323 | **-0.00191** | -0.00205 | -0.00041 |
| PK-BPO (-known) | 0.14351 | **-0.00071** | -0.00141 | -0.00438 |

**Every arm is negative.** The 2-6% precision cannot survive the `prop=fill` tax (each false extra
forfeits its ancestors' free inheritance). The channel does not convert.

## Leakage -- CLEAN
- **Edges:** STRING v12.0 static release 2023-07-26 << t0 2025-09-04. `textmining` never read in the
  headline; `database` only in the flagged ablation.
- **Partner annotations:** ONLY the frozen v227=t0 `reference_annotations.parquet` (cutoff 2025-09-04,
  strictly before the v227->v230 eval gains) -- the same artifact CG.1 certified clean. Self and
  same-accession partners excluded.
- **Empirical target-as-partner ablation** (analog of CG.1's "0 BP in LK known"): removing every
  partner that is itself a test target keeps **76% (LK) / 91% (PK)** of the added-true mass and precision
  stays above the floor (LK 0.059 -> 0.0414; PK 0.0252 -> 0.0199). The signal is carried by
  **independent, separately-annotated partners**, not target-to-target laundering. Partner annotations
  are t0-frozen and the target gains are post-t0, so there is no temporal leakage by construction.

## Implication
Five generation channels are now characterised: sequence co-occurrence (CG.1), literature (CG.2), the
full-GO classifier, and now the PPI/co-expression network (CG.3). The last untested orthogonal axis
behaves like the rest -- the missing true BP tail is **reachable but not separable**. **The BP
separability wall is a field frontier, not an in-house engineering gap.** The network hypothesis
(process = who you interact / co-express with) is directionally correct -- network partners are the best
generator we have -- but the candidates it proposes are true only 2-6% of the time, which the metric
cannot exploit.

## Receipts (all under `storage/regen_headline/`)
- `cg3_step1_gate.py` / `.json` -- the gate (both arms, per-k, stratification).
- `cg3_step2_cafaeval.py` / `.json` -- the true-frame board delta.
- `cg3_leakage_ablation.py` / `.json` -- the target-as-partner leakage ablation.
- `cg3_verdict.json` -- machine-readable verdict.
- `CG3_NETWORK_CHANNEL_PREP.md` -- the pre-download design (feasibility, resource spec, coverage).
- `storage/string_v12/` -- 18 tier-1 STRING v12.0 per-species files (1.7 GB).
