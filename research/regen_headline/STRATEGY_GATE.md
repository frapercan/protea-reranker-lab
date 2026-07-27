# The reranker is ignoring its best signal on PK, and the classifier alone beats it

Receipt for the per-cell strategy gate. Script `storage/cooc_experiment/strategy_gate_crossfit.py`,
data `strategy_gate_crossfit.json`.

## The result

Three complete scorers, each submitting its own rows where its own score is strictly positive:
the deployed booster's margin, the raw neighbour vote fraction, and the sparse two-tower logit
alone. Ten disjoint protein folds per cell; the best arm is chosen on the other nine pooled and
applied blind to the tenth. Scored per cell against that cell's own board ground truth.

| cell | arm chosen | held-out delta vs reranker | sd across folds | positive | gate |
|---|---|---|---|---|---|
| **PK-BP** | **classifier alone** | **+0.02106** | 0.00789 | **10/10** | **PASS** |
| LK-BP | classifier | +0.02746 | 0.04043 | 7/10 | FAIL |
| NK-BP | reranker | 0.0 | 0.0 | 0/10 | FAIL |

**On PK-BP the sparse classifier alone beats the whole deployed reranker by +0.021 held-out, and
all ten folds chose it without seeing the fold they were scored on.** That is more than twice the
prefilter lever. The precondition held in every cell: fold-level `reranker` reproduced the
whole-cell anchors from `build_submission.py` (PK 0.20316 vs 0.21269, LK 0.3054 vs 0.30376, NK
0.31106 vs 0.3021).

## Why this is not a surprise, and why nobody saw it

The nine-cell LOFO already recorded that the classifier family contributes **exactly zero** on PK:
dropping it changes nothing, because the reranker does not use it there. That was read as "the
classifier is useless on PK". It means the opposite. The signal the reranker discards on PK is
**better than everything the reranker produces on PK**.

It joins the two coverage findings that were already on the table. The classifier proposes 43.4%
of the pk-bpo pool and those candidates carry **78% of the positives**, and the KNN-path producers
never ran over them, so `length_query`, `distance` and every neighbour feature are NaN on 100% of
those rows. The reranker cannot rank what it cannot see, so on PK it falls back on the KNN half and
throws away the half that holds most of the truth. The classifier, scoring its own candidates with
its own logit, keeps them.

## LK-BP failed the gate, and the failure is the point

The classifier looks better on LK too, and by a larger mean (+0.02746), and it is **not
reportable**: the fold-to-fold sd is 0.04043 and only 7 of 10 folds are positive. The mean does not
clear the noise. LK-BP has 523 ground-truth proteins, so a fold is 52 proteins and the variance
swallows the effect. Without the gate this would have shipped as a second win. It is not one.

NK-BP is the trivial case: the reranker wins in all ten folds, the delta is identically zero
because the chosen arm is the incumbent, and the gate fails by construction. Nothing to change.
Worth noting for any future NK claim: fold-level f ranges from 0.209 to 0.423 across ten folds of
27 proteins. **NK-BP cannot support a claim without an interval.**

## What it corrects

Lab #113 recommended "rerank NK/LK, raw-KNN PK" and its TEST-frame export never ran. It was right
that PK wants a different scorer and wrong about which one: **raw KNN was never chosen in any fold
of any cell.** The classifier was.

## The frame, stated because it is binding

Lab frame: `protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo` declares `data-version:
releases/2025-07-22`, and the board's t0 is September 2025 on the host's cluster. **Every arm here
shares one obo and one IA, so the reweighting is common to all of them and cancels in the
comparison.** This is a within-frame delta and it is legitimate as such. **No number above is a
board number**, and +0.021 must not be added to 0.2181 to advertise a board figure.

## What it does not do

It does not flip PK-BPO. TransFew sits at 0.2943 and the gap is far wider than this. It is the
largest gated gain of the campaign and it is still not enough on its own.

**Untested and next:** whether this composes with the prefilter (+0.0092). They may overlap
entirely, since both act by withholding weak candidates, and the honest way to find out is one
variable at a time, cross-fitted, with the gate written first.

---

# PKW.10: the levers compose, and one of my own results does not survive

Script `storage/cooc_experiment/compose_prefilter_classifier.py`, data
`compose_prefilter_classifier.json`. Four arms on PK-BP, one variable at a time, ten disjoint
protein folds, every threshold chosen on nine pooled and applied blind to the tenth, all on the
board's real line.

**The precondition came back exact: arm A's fold-mean is 0.20316 against an anchor of 0.20316.**

| comparison | mean | sd | positive | |
|---|---|---|---|---|
| **B - A**  prefilter on the **reranker** | **-0.00032** | 0.00581 | 7/10 | **flat** |
| **C - A**  classifier alone | **+0.02106** | 0.00789 | **10/10** | reproduces the gate exactly |
| **D - C**  prefilter on the **classifier** | **+0.00759** | 0.00283 | **10/10** | **PASS**, tau=0.776 unanimous |

**The levers compose: classifier plus its own prefilter is +0.0287 over the reranker**, and both
components are unanimous across ten folds. Fold-mean D is 0.2314 against A's 0.20316.

## The contradiction, stated rather than resolved

**PKW.8 measured the prefilter on the reranker at +0.0092, unanimous across the same ten folds.
Here the same lever is flat: -0.00032, positive in 7 of 10.** These are two measurements of one
thing by the same author on the same data and they disagree.

Two things differ between the runs and neither has been isolated: PKW.8 submitted the **bpo-only**
rows of `eval_scores.parquet` and scored them on the **lab's** line (`max_terms=500`,
`th_step=0.001`, no `-toi`, i.e. the legacy parser); this run submits **all aspects** of the pk
rows, board-shaped, on the **board's** line (`max_terms=None`, `th_step=0.01`, `-toi`, i.e. the
vectorised parser). `build_submission.py` sits in between and agrees with PKW.8 (+0.0102) at
whole-cell level on the board's line, which rules out the flags being the whole story and leaves
whole-cell versus fold-level as a live suspect, since f_micro_w is a micro average and is not the
mean of per-fold f.

**I cannot explain it in this run and I am not going to pick the version I prefer.** The prefilter
lever on the reranker is **open**, not confirmed and not dead, and it needs the one-variable test
that was never run: same rows, same folds, one flag moved at a time.

What is unaffected: **C - A** reproduces the strategy gate to the digit (+0.02106, sd 0.00789,
10/10), and **D - C** is a fresh, gated, unanimous result that never depended on PKW.8.

## The asymmetry, which explains the mechanism from the inside

**Prefiltering the reranker does nothing; prefiltering the classifier works.** That fits what we
proved: the reranker's weak candidates already score at or below zero, so the parser discards them
for free and a cut at 0.4 mostly removes real signal. The classifier's weak candidates score as
small **positive** logits, so they are stored, and `fill` protects them, blocking the ancestor
inheritance. Cutting those is exactly the act the mechanism rewards, and the unanimous choice of
tau=0.776, the 60th percentile of the classifier's own positive logits, is where it stops paying.

It is a tidy story, it agrees with the proven mechanism, and tonight five such stories have already
been false. It is recorded as coherent, not as established, and it is not what the gate tested.

## What it still does not do

It does not flip PK-BPO. Fold-mean 0.2314 against TransFew's 0.2943, and that comparison is
between frames in any case. **No number here is a board number.**
