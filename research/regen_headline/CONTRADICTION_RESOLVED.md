# The contradiction is resolved: the bug was mine, and it was the one I had already catalogued

Script `storage/cooc_experiment/resolve_contradiction.py`, data `resolve_contradiction.json`.

## The contradiction

The same lever (prefilter the reranker at tau_pre=0.4), the same ten folds, the same seed,
measured twice by me:

- PKW.8 (`prefilter_crossfit10.py`): **+0.0092**, sd 0.0034, unanimous 10/10.
- PKW.10 (`compose_prefilter_classifier.py`, arm B minus arm A): **-0.00032**, 7/10.

Four things differed and none had been isolated. The ladder changes exactly one per rung.

| rung | config | prefilter delta | step |
|---|---|---|---|
| **L0** | PKW.8: bpo-only rows, aligned join, lab line (`max_terms=500`, `th_step=0.001`, no `-toi`) | **+0.00924** (sd 0.0034, 10/10) | |
| **L1** | + the board's real line (`max_terms=None` so the vectorised parser, `th_step=0.01`, `-toi`) | +0.00922 | **-0.00002** |
| **L2** | + all-aspect pk rows instead of bpo-only | +0.00922 | **+0.00000** |
| **L3** | + the dict join | **-0.00032** (7/10) | **-0.00954** |

**The dict join is the entire cause.** The flags move two hundred-thousandths. The row set moves
nothing at all, exactly zero. The fourth rung takes the whole effect.

## The bug

PKW.10 and the strategy gate both built `RRmap = dict(zip(zip(prot, go), score))`. The pk-bpo pool
holds **616,223 rows over 576,031 distinct (protein, term) keys: 40,192 are duplicated**, emitted
once by the KNN generator and once by the classifier. A dict keyed on (protein, term) is
**last-wins**, so it silently rewrote **71,574 of 863,748 rows, 8.3%**.

They are not copies. The two scores of a duplicated pair differ by a **median of 3.60** and by up
to 9.00, and for **9,510 pairs the two straddle zero**, so the dict decided by row order whether
that candidate was submitted at all. PKW.8 read the aligned arrays and every row kept its own
score. The two tables are row-for-row identical, which `strategy_gate_aligned.py` now asserts
rather than assumes, so the dict was never needed in the first place.

**This is the exact bug catalogued last night in `is_the_reranker_hurting_pk.py` (keyed on
non-unique (protein, term); 40,192 pairs collapsed). I wrote it down, and then I wrote it again
twelve hours later.** Cataloguing a bug does not prevent it. A row-alignment assertion does.

## What this settles

**PKW.8 stands, and stronger than before.** The prefilter on the reranker is **+0.0092, unanimous
across ten folds**, and it is now shown to survive the board's real line unchanged (L1) and a
board-shaped all-aspect submission unchanged (L2). The doubt I raised was mine, not the result's.

**PKW.10's arms A and B are void**, and with them **C - A = +0.02106**: it compared the classifier
against a corrupted reranker. **The strategy gate used the same join, so "the classifier alone
beats the reranker on PK-BP by +0.021", which I announced as the biggest gated gain of the
campaign, is not established at that magnitude.** It is being re-run aligned
(`strategy_gate_aligned.py`).

**D - C = +0.00759 survives** on both counts: `classifier_score` was read aligned from
`eval.parquet` and never passed through the dict, so both of its arms were clean.

## Why the precondition did not catch it

The precondition required arm A's fold-mean to land within **0.02** of the anchor, while the effect
under measurement was **0.021**. It passed a corrupted arm and reported the design sound.

**#16: a precondition's tolerance must be tighter than the effect it is protecting.** A tolerance
wider than the effect tests nothing and buys false confidence, which is worse than no check at all.

---

## The strategy gate, re-run with a verified aligned join

`storage/cooc_experiment/strategy_gate_aligned.py`, data `strategy_gate_aligned.json`. Identical
design, identical folds and seed; the only change is the join, and the script now **asserts** the
two tables are row-for-row identical instead of trusting it.

| cell | arm | aligned delta | sd | positive | gate | previously announced |
|---|---|---|---|---|---|---|
| **PK-BP** | classifier | **+0.00839** | 0.00578 | **9/10** | **PASS, barely** | +0.02106, 10/10 |
| LK-BP | classifier | +0.0273 | 0.04124 | 7/10 | FAIL | unchanged |
| NK-BP | reranker | 0.0 | 0.0 | 0/10 | FAIL | unchanged |

**The dict was suppressing the reranker by 0.013** (fold-mean 0.20316 corrupted against 0.21584
aligned), and that suppression was most of the classifier's apparent advantage.

So the corrected result: **the classifier alone still beats the reranker on PK-BP and the gate
still passes, but by +0.0084, not +0.021.** Less than half. And it passes narrowly: 9 of 10 folds
rather than 10, and the mean clears the sd by 1.45x rather than 2.7x.

## The map, corrected

| lever | delta | sd | positive | mean/sd |
|---|---|---|---|---|
| **prefilter the reranker (PKW.8)** | **+0.0092** | 0.0034 | **10/10** | **2.7x** |
| classifier alone on PK (aligned) | +0.0084 | 0.00578 | 9/10 | 1.45x |
| prefilter the classifier (D - C) | +0.00759 | 0.00283 | 10/10 | 2.7x |

**The most robust lever in the campaign is the prefilter**, which I demoted to "open" last night on
the strength of my own bug, and then crowned the classifier using the same bug. The three levers
are now the same size, and whether they compose is once again an open question: PKW.10's D - C was
measured against a clean C but its relationship to a clean A was never established.

**Nothing here flips PK-BPO.** All of it is within-frame and none of it is a board number.

---

## Everything reconciles once the join is aligned

`storage/cooc_experiment/compose_aligned.py`, data `compose_aligned.json`. Same four arms, same
folds, aligned and asserted join, and a precondition tolerance of **0.004**, tighter than the
~0.008 effect it protects (the old 0.02 was wider than the 0.021 it was supposed to guard, which
is how the corrupted arm walked through).

**Precondition exact: arm A's fold-mean is 0.21584 against an anchor of 0.21584.**

| comparison | mean | sd | positive | |
|---|---|---|---|---|
| **B - A**  prefilter on the reranker | **+0.00922** | 0.00374 | **10/10** | reproduces PKW.8 to the digit |
| **C - A**  classifier alone | **+0.00839** | 0.00578 | 9/10 | reproduces `strategy_gate_aligned.py` |
| **D - C**  prefilter on the classifier | **+0.00759** | 0.00283 | **10/10** | tau=0.776 unanimous |

**The levers compose.** Three independent measurements now agree with each other, which they could
not do while the dict was in the way.

Best arm: **D, the classifier with its own prefilter, at a fold-mean of 0.23182 against the
deployed reranker's 0.21584: +0.016.**

The pattern worth keeping: **the prefilter works on both scorers**, +0.0092 on the reranker and
+0.0076 on the classifier. It is not a quirk of one model, it is a property of how `prop=fill`
scores a submission: withhold a weak candidate and its cell inherits its best descendant for free;
submit it and that inheritance is blocked.

It does not flip PK-BPO. 0.232 against TransFew's 0.2943, and across frames at that. What it does
give, for the first time in the campaign, is **three levers that are measured, gated, and mutually
consistent.**
