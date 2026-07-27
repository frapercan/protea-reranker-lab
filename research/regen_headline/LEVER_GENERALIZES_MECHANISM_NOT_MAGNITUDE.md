# The generation lever's MECHANISM generalizes to LK-BPO; its MAGNITUDE does not

`storage/cooc_experiment/score_the_extras_lk.py` (+ `.json`, `.log`)

## The question

The +0.02245 classifier-extras generation lever is confirmed on PK-BP. Does it transfer to the other
losing cell, LK-BP? Same machinery: arm A = deployed pool (anchor legacy 0.30376, no prefilter), arm
B = pool + classifier extras scored by a transferred S, arm C = the same extras at the raw sigmoid,
swept over top-k, scored in the legacy (campaign) frame.

**CAVEAT, load-bearing:** the local lk-bpo baseline is DEGRADED. It reads 0.30376 locally against the
board's cited 0.4402, because the deployed LK arm carried ProtST/abstract signals absent from this
parquet (`WE_DO_NOT_REPRODUCE_THE_BOARD`). So no number here transfers to the board; the question is
purely whether generation GENERALIZES as a mechanism.

## The result

Precondition PASSED: arm A = 0.30376, reproducing the local deployed anchor to the digit.

| top-k | extras | B (scored by S) | C (raw sigmoid) |
|---|---|---|---|
| 20 | 2,772 | +0.00148 | -0.00497 |
| 50 | 11,630 | +0.00374 | -0.00807 |
| 100 | 30,957 | +0.00306 | -0.00865 |
| 200 | 74,902 | **+0.00387** | -0.00876 |

## What it says

**The mechanism generalizes exactly.** The PK signature reproduces on LK: S beats the raw sigmoid at
every k, and C (raw sigmoid) loses monotonically as k grows because under `prop=fill` every unscored
false extra forfeits its ancestor's free inheritance. So the lever is not a PK accident: generation
plus a transferred scorer is a general channel, and the scoring is what makes the extras pay.

**The magnitude does not.** Best B is +0.00387, barely above the 0.0034 fold noise (~1.1x) and ~6x
smaller than PK's +0.02245. This is a SINGLE split with no interval, so +0.00387 is NOT a confident
positive; it is marginal and mechanism-consistent, not a confirmed lever. The reason is structural:
LK proteins already carry prior knowledge, so there is far less room for new candidates than in the
prior-knowledge-free PK regime, which is exactly why the classifier-alone arm failed the LK gate
(sd 0.040, 7/10). The cross-aspect atlas said the same: we lose where the protein already knows
something.

## For the thesis

This STRENGTHENS the generation story without overclaiming: the lever's mechanism (generation scored
by a transferred model, beating the raw sigmoid against the fill tax) is general across BP, and its
payoff is concentrated in PK where the headroom lives. Generation is THE channel; the deployed
reranker's scoring/objective/architecture/representation are all near-optimal (measured negatives).
The BP wall is a generation problem, and generation's value is PK-dominant.
