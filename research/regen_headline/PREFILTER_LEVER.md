# The prefilter threshold: the first free gain of the campaign

Receipt for PKW.8. Scripts: `storage/cooc_experiment/prefilter_crossfit.py` (2-fold),
`storage/cooc_experiment/prefilter_crossfit10.py` (10-fold, the one that decides).
Data: `prefilter_crossfit.json`, `prefilter_crossfit10.json`.

## The mechanism, and it is observable from outside

`prop="fill"` overwrites only cells that are EXACTLY ZERO (`graph.py:320-323`), and the
parser's write guard `if prob_f > old` from `old=0.0` (`parser.py:230`) discards every
non-positive score. A candidate you do not submit therefore becomes a zero cell, and `fill`
overwrites it with its best descendant's score: a free ancestor, arriving at weighted
precision 0.2967 against the arm's own 0.1801. A candidate you submit with a low score is
stored, and `fill` protects it, blocking the inheritance.

**Submitting a bad candidate is worse than not submitting it. It robs a good ancestor.**

The sweep confirms this from outside the code. On both folds, every threshold from -2.0 to
0.0 gives an identical f to four decimals while the submitted rows fall from 185,070 to
57,784. Submitting a negative score and withholding it are the same act.

| tau_pre | fold A f | fold B f | rows (A) |
|---|---|---|---|
| -2.0 | 0.2121 | 0.2161 | 185,070 |
| 0.0 (the accidental default) | 0.2121 | 0.2161 | 57,784 |
| 0.2 | 0.2182 | 0.2235 | 34,814 |
| **0.4** | **0.2207** | **0.2270** | 19,276 |
| 0.6 | 0.2041 | 0.2162 | 9,636 |
| 1.0 | 0.1229 | 0.1203 | 1,936 |

## The result

The published anchor 0.2131 comes from lambdarank margins running [-8.48, 3.81] and the
clamp silently dropping the 80.9% at or below zero. **Nobody chose that threshold.** It is a
free parameter that arrived by accident, and it is not the optimum.

Ten disjoint protein folds. For each, the threshold was swept on the other nine pooled and
applied blind to the tenth. Board flags exactly: `prop=fill, norm=cafa, no_orphans,
max_terms=500, th_step=0.001`.

- **All ten folds independently chose tau_pre = 0.4.** Unanimous.
- **Held-out delta positive in 10 of 10 folds.** Mean **+0.0092**, sd across folds 0.0034.
- Mean exceeds one standard deviation by a factor of 2.7.

Both gate conditions were met: the selections agree, and the gain sits outside the noise.

## What this does not do

**It does not flip PK-BPO.** The lab anchor 0.2131 corresponds to the board's 0.2181, so
0.2223 in the lab frame lands near 0.227 on the board against TransFew's 0.2943. The cell
stays lost. This is a real gain and a small one.

**It measures generalisation across proteins, not across time.** Selecting on the validation
window and applying blind to test is impossible on frozen data: there is no full ground truth
for v225-v227 (the lab's `gt_pk_bp.tsv` is the test gt, and a validation gt rebuilt from
`label=1` rows would be pool-restricted, inflating recall and biasing the very threshold
being chosen), and the deployed booster early-stopped on that window. A threshold that
transfers between protein folds may still drift between snapshot windows. This cannot see
that, and does not claim to.

## A gate I wrote wrong, recorded because the record is the point

`prefilter_crossfit.py` wrote its gate before the number, as the discipline demands, and the
gate was malformed: it defined the selection spread as max-min of f over the whole grid. That
grid deliberately includes thresholds that destroy information (tau_pre=1.5 submits 177 rows
and scores 0.026), so max-min measures the shape of the curve, not the noise in picking a
point on it. No result could ever have passed it. It is reported there as False and it means
nothing. `prefilter_crossfit10.py` replaces it with a real noise estimate: the spread of the
held-out delta across ten independent selections. **Writing the gate first does not protect
you if the gate measures the wrong quantity.**

## Ground truth, verified rather than assumed

The board's own ground truth is `CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_{NK,LK,PK}.tsv`.
`groundtruth_PK.tsv` aspect P is 41,727 pairs over 4,402 proteins: **exactly** the lab's
`gt_pk_bp.tsv`, with zero pairs on either side of the difference. The lab and the board agree
on what is true. The 0.2131 / 0.2181 gap is therefore in the predictions, not the truth.

Cell sizes, which explain the geography of the problem: NK-BP has 267 proteins, LK-BP 523,
PK-BP 4,402. The two cells we lose are not the large ones.

## An open receipt gap, found here

The board's best result for us is `predictions_protea.tsv` at f_micro_w 0.21807 with an
optimal tau of 0.06, so the submitted file carries a probability-like scale, not lambdarank
margins. **That file does not exist anywhere in the tree.** The only local submission is
`predictions_percutgraft.tsv` (in [0,1], 11.25% exactly zero), which the board scores at
0.140 and which is a different, worse arm. The 0.2181 we quote as ours is a board-side
artifact we cannot currently regenerate, and the prefilter above cannot be applied to the
submission until it can be. That is the next thing to fix, and it is a precondition for
shipping this lever, not a detail.

---

# Second confirmation: the lever holds on a real submission file

Script `storage/regen_headline/withhold_on_real_submission.py`, data
`withhold_on_real_submission.json`. Everything above was measured on `eval_scores.parquet`, our
own intermediate. This uses `results/clean_227230/lafa_submission/predictions_7401_reranked.tsv`
(27 June, 464,785 rows over 7,401 proteins, scores in [0.0001, 1.0], **zero exact zeros**), a file
built to be sent, scored under the board's real line.

| withhold bottom | rows | f_micro_w | pr | rc |
|---|---|---|---|---|
| 0% (as built) | 464,785 | 0.20132 | 0.1698 | 0.2473 |
| 20% | 371,789 | 0.21469 | 0.1762 | 0.2746 |
| 50% | 232,238 | 0.21539 | 0.1764 | 0.2764 |
| **70%** | 139,321 | **0.21580** | 0.1764 | 0.2779 |
| 90% | 46,413 | 0.21545 | 0.1921 | 0.2452 |

**+0.0145 for withholding the bottom 70%**, against the 0.0034 fold-to-fold sd from the ten-fold
cross-fit. A different arm, a different score scale, a different file, and the same mechanism:
this submission gives every candidate a strictly positive score, so under `prop=fill` no cell is
ever the exact zero that inherits a free ancestor, and it pays full price for its own worst
candidates. Notice precision barely moves while recall rises: the withheld cells come back as
ancestors, which is the mechanism stated in its own terms.

## The precondition gate fired, and it was right to

The gate required arm A to reproduce the run recorded beside the file (PK-BP 0.117) before
anything below it could be trusted. **It did not: A scores 0.20256 under the recorded flags.**
Not a flag difference: `tau` matches (0.522 vs 0.520) and `cov_w` matches (0.8512 vs 0.853), so
the same file sits at the same operating point. The recorded row evaluates **n = 3,756**
proteins; the board's `groundtruth_PK.tsv` carries **4,402** in BP. **The recorded run used a
different ground truth**, so 0.117 and 0.2026 were never comparable quantities.

## A second tidy story of mine, killed by the same reflex

On seeing 0.117 I built a four-arm ladder: `predictions_7401_reranked` 0% zeros gives 0.117,
`predictions_percutgraft` 11.25% zeros gives 0.140, `predictions_protea` gives 0.2181, our
prefilter gives 0.22288, therefore the withheld fraction predicts the score. It was neat, it
matched the mechanism we had already proven, and it was **wrong**: under the board's line that
first arm scores 0.20132, not 0.117, and the ladder does not exist. The 0.117 was an artifact of
a different ground truth.

Twice in one run a tidy account fitted the evidence and was false. Both times the tell was the
same: the story explained the numbers without anyone checking that the numbers were commensurable.
**#12: before comparing two numbers, prove they measure the same thing.** `tau` and `cov_w`
agreeing while `f` disagreed was the signal, and `n` was where it was written down.
