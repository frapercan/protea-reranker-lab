# Cross-modality consensus and the BP separability wall

**Verdict: NO-GO.** Cross-modality agreement *does* compound precision (the independent-error
mechanism the author predicted is real, ~20x from 1 to 3 modalities), but the high-precision
intersection tops out at ~2.5-3% precision, four times below the ~12% breakeven where the
classifier lever already died, and every temporally-gated augmentation of the deployed submission
**loses** with a bootstrap CI entirely below zero. The intersection is not too small (it retains
27-38% of the available true IA-mass); it is not precise enough. The BP tail does not separate
under three orthogonal channels.

Read-only, frozen data, CPU, no live DB, no job dispatch. Classifier logits recomputed from the
cached `generator_frames` (no DB). Phylogenetic profiling was **pending** (no proposals; the ODB12
download was still running under `storage/phylo_profile/`), so this ran with **3 orthogonal
modalities**: sequence-kNN (the 6-PLM family as ONE channel), the STRING network, and the full-BP
classifier.

## 1. The consensus precision curve (Phase 0, full board frame Sep_2025->Mar_2026)

For BP candidates NOT already in the deployed pool (PK also drops `-known`), precision and true
IA-mass as a function of the number of ORTHOGONAL modalities agreeing. Each modality is recall-heavy
by construction (so intersections are non-empty): seq-union median ~1138 BP terms/protein, STRING
~267, classifier top-500. Single-channel bars for reference: co-occ 1%, network 5.8%, classifier
11.6% (those bars were measured at small high-precision operating points, e.g. classifier top-5;
here every channel is run recall-heavy, which is what a consensus intersection requires).

**LK-BPO** (495 proteins, 690,707 candidates, 3,150 total true IA available):

| # modalities | n cand | precision (raw) | precision (IA-weighted) | true IA-mass | % of available true IA |
|---|---|---|---|---|---|
| agree = 1 | 534,327 | 0.11% | 0.11% | 814 | 26% |
| agree = 2 | 123,932 | 0.93% | 0.85% | 1,499 | 48% |
| agree = 3 | 32,448 | **2.57%** | **2.06%** | 838 | 27% |
| agree >= 2 | 156,380 | 1.27% | 1.08% | 2,337 | 74% |

**PK-BPO** (4,349 proteins, 6,181,267 candidates, 36,600 total true IA available):

| # modalities | n cand | precision (raw) | precision (IA-weighted) | true IA-mass | % of available true IA |
|---|---|---|---|---|---|
| agree = 1 | 4,496,780 | 0.14% | 0.12% | 8,029 | 22% |
| agree = 2 | 1,235,735 | 0.99% | 0.87% | 14,777 | 40% |
| agree = 3 | 448,752 | **3.10%** | **2.45%** | 13,794 | 38% |
| agree >= 2 | 1,684,487 | 1.55% | 1.26% | 28,570 | 78% |

**Does cross-modality agreement compound precision?** Yes, monotonically and steeply: each added
orthogonal channel multiplies precision ~7-8x per step (LK 0.11 -> 0.93 -> 2.57%; PK 0.14 -> 0.99 ->
3.10%), a ~23x total lift from 1 to 3 modalities. This is exactly the independent-error compounding
the hypothesis predicts, and it distinguishes cross-modality agreement from within-PLM agreement
(where errors are correlated). The mechanism is confirmed.

**Is the ceiling high enough?** No. The 3-way intersection (2.5-3.1% raw, ~2.0-2.5% IA-weighted)
sits *below the network single-channel bar (5.8%)* and roughly **4x below the ~12% breakeven**: the
same classifier generator at 11.6% precision was already measured at **-0.013** f_micro_w under
`-known` (`score_the_extras_trueframe`, top100). A 2.5-3% candidate set cannot clear the prop=fill
tax. The IA-mass is not the bottleneck (agree>=3 keeps 27% LK / 38% PK of the available true IA, and
agree>=2 keeps ~75%); the precision is.

## 2. Does any consensus threshold clear the fill breakeven while keeping IA-mass?

**No.** The best precision at any threshold is agree=3 at 2.5-3.1% raw / 2.0-2.5% IA-weighted.
agree>=2 is even lower (1.1-1.3% IA-weighted). Both are far under the empirical breakeven (~12%+,
below which even the isolated classifier generator loses). No threshold reaches a precision that
plausibly clears the tax, so the precision gate fails.

## 3. Confirmatory temporally-gated f_micro_w (Phase 1)

Even though the gate failed, the decision was measured in the decisive metric via the mandated
temporal-gate harness (adapted verbatim from `multiplm_pool.py`): fit a GBM precision policy +
choose phi/tau on Sep_2025->Nov_2025 BLIND, apply the FROZEN policy on Nov_2025->Mar_2026, augment
the DEPLOYED submission, score f_micro_w true-frame. Anchor = the deployed percutgraft submission
recomputed on the APPLY window (NOT the raw pool). Matched-volume uniform control + paired protein
bootstrap (B=2000). cafaeval parity held on every arm.

| cell | FIT phi | extras added | extra precision (apply) | mean consensus level | deployed (apply-win) | consensus (apply-win) | **delta vs deployed** | bootstrap CI | vs uniform |
|---|---|---|---|---|---|---|---|---|---|
| LK-BPO | 0.002 | 453 | 6.4% | 2.55 | 0.28051 | 0.25332 | **-0.02719** | [-0.044, -0.012] | +0.072 |
| PK-BPO | 0.005 | 12,124 | 5.0% | 2.39 | 0.16244 | 0.09719 | **-0.06525** | [-0.077, -0.053] | +0.052 |

Both deltas vs the deployed anchor are **negative with a bootstrap CI entirely below zero**
(fraction of positive bootstrap draws 0.0005 LK / 0.0000 PK). The consensus selection *does* beat
the matched-volume uniform random control (+0.072 / +0.052), so the policy is selecting genuine
signal, not noise, but that signal is still far too dilute: adding it *removes* f_micro_w because
each low-precision extra forfeits its ancestors' free inheritance under prop=fill. The FIT window
looked spuriously excellent (+0.057 LK, +0.197 PK), which is exactly the fit-window overfitting trap
these gates exist to catch; the blind window collapses it. This reproduces the within-PLM
multi-agreement result (`MULTIPLM_POOL`, which lost -0.027 LK) and extends it: **orthogonal-channel
consensus is no better than correlated-channel agreement at the board metric.**

**Frame sanity (precondition, PASSED):** the full-window deployed anchor reproduced the canonical
values **exactly** to 5 decimals: LK-BPO 0.31323, PK-BPO 0.14351 (`fullwindow_anchor_reproduction`).
No wrong-anchor artefact. The apply-window anchors (0.28051 / 0.16244) are the correct paired
controls for the Nov->Mar sub-window; the delta is what decides, and it is negative.

**Leakage (CLEAN):** all three channels' transferred annotations are strictly t0. seq-kNN = v227
homolog transfers, self-neighbour dropped; STRING v12.0 (2023); classifier trained on v227
experimental labels, logits from cached t0 frames. The candidate LABEL is the only post-t0 object.
Temporal gate: policy sees only <=Nov_2025 labels + t0 strata, applied blind to Nov->Mar.

## One-line verdict

**NO-GO.** Cross-modality consensus compounds precision as predicted (~23x, mechanism real) but the
high-precision intersection peaks at ~2.5-3% (IA-weighted ~2-2.5%), 4x below breakeven; every
temporally-gated augmentation loses vs the deployed 0.14351/0.31323 anchor (LK -0.027 CI
[-0.044,-0.012]; PK -0.065 CI [-0.077,-0.053]). Three orthogonal channels do not separate the BP
tail: the intersection is high-purity relative to each channel yet still too impure to pay the
prop=fill tax, and it is not small (keeps 27-38% of true IA-mass), so the wall is separability, not
reachability.

## Receipts

- Precision curve: `storage/consensus/phase0_precision_curve.py` -> `phase0_precision_curve.json`
- Temporal gate: `storage/consensus/phase1_temporal_gate.py` -> `phase1_temporal_gate.json` (log `phase1.log`)
- Cached classifier proposals (no DB): `storage/consensus/clf_prop_topk.pkl`
- Deployed anchor: `repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/predictions/`
- Phylo (4th modality) was pending at run time: `storage/phylo_profile/` (no proposal pkl).
