# 0.4063 is not derivable from the board. The board's own nine cells average 0.40765.

Script `storage/regen_headline/rederive_0_4063.py`, data `rederive_0_4063.json`.

## What was tested

Six candidates, **all declared in the script before any was computed**, because the way to fail
this question is to keep trying definitions until one agrees.

| candidate | value | delta vs 0.4063 |
|---|---|---|
| **unweighted mean of the nine current cells** | **0.40765** | **+0.00135** |
| weighted by ground-truth protein count | 0.27331 | -0.13299 |
| weighted by ground-truth pair count | 0.28587 | -0.12043 |
| unweighted mean of `f_micro` rather than `f_micro_w` | 0.46015 | +0.05385 |
| each `.bak_*` board state | no cells | our arm is not in them |
| the other five release windows on disk | no cells | our arm is not in them |

The nine cells, cited from the board's frozen table:

| | MF | BP | CC |
|---|---|---|---|
| **NK** | 0.6637 | 0.3374 | 0.4770 |
| **LK** | 0.5638 | 0.4402 | 0.4615 |
| **PK** | 0.2416 | 0.2181 | 0.2655 |

## The result

**`predictions_protea.tsv` appears in exactly one board state, the current one, and its nine cells
average 0.40765.** The board keeps three earlier states (`.bak_prereranked`, `.bak_pregraft`,
`.bak_preinterpro`) and five other release windows, and our arm is in none of them: the
`prereranked` state carries `protea-knn-v1.tsv` and no `predictions_protea.tsv` at all, so the
"0.4063 came from an earlier board" hypothesis is closed by data rather than by argument.

**No declared candidate reproduces 0.4063.** 0.4063 x 9 = 3.6567 against the current sum of
3.66885, a difference of 0.01215 spread across the nine, which is the size of one cell moving
slightly rather than of a different statistic.

## The decision, which is the author's

The sealed headline is **0.4063**. The board's own nine cells average **0.40765**. The difference
is **+0.0013 in our favour**, and the higher figure is the one with a script behind it.

Three readings are available and this note does not pick between them:

1. **0.4063 was computed from a board snapshot that no longer exists.** The board has been recomputed
   at least three times; the headline may date from a state between two of the saved ones.
2. **0.4063 is a transcription of 0.40765 that lost precision somewhere** and then hardened.
3. **0.4063 is a different statistic nobody wrote down.** Four were declared and tested; a fifth
   exists in someone's memory and not in the tree.

**Nothing here is wrong on the surfaces today**: 0.4063 understates the board by 0.0013 and every
cell it averages is cited correctly. This is a provenance gap, not an error, and the fix is a
decision rather than a correction: **publish 0.40765 with `rederive_0_4063.py` beside it, or keep
0.4063 and record which snapshot produced it.** Until the author chooses, the seal holds.

## Frame

The nine cells are the board's own figures, cited from its frozen table, so any mean of them stays
in the board's frame. This is the one family of absolute numbers we may quote, and the rule that
makes it safe is in `WE_DO_NOT_REPRODUCE_THE_BOARD.md`.

---

# Resolved: 0.4063 is an offline projection. 0.40765 is what the board recorded.

The origin was in the code, not in the receipts. Two docstrings name both the statistic and the
arm:

- `protea/core/operations/_run_cafa_interpro_graft.py:17`: "the shipped ``naivemax_bponly`` graft
  that lifts the **board-faithful 9-cell mean** ``f_micro_w`` from **0.3884** to **0.4063**".
- `protea/core/operations/run_cafa_evaluation.py:110`: "Reproduces the offline ``naivemax_bponly``
  champion (board-faithful 9-cell mean ``f_micro_w`` 0.3884 -> 0.4063)".

So the statistic was never in doubt: it is the unweighted nine-cell mean, exactly the candidate
that lands at 0.40765. What was in doubt is which run produced 0.4063, and the answer is that no
run did. Computing every arm's nine-cell mean from the board's own frozen table:

| arm | board 9-cell mean |
|---|---|
| **`predictions_protea.tsv`** (ours, the current champion) | **0.40765** |
| `predictions_percutgraft.tsv` | **0.38844** |
| `transfew_predictions.tsv` | 0.38116 |
| `funbind_predictions.tsv` | 0.36599 |
| `predictions_7401_reranked.tsv` | 0.35933 |
| `protea-knn-v1.tsv` | 0.31156 |
| `naive_predictions.tsv` | 0.11020 |

**0.38844 is the code's 0.3884, to the digit.** The pair the docstrings quote is therefore a real
measurement of one arm and an offline projection of the other: the lab computed what the graft
would be worth and wrote 0.4063 down, the board then scored the grafted arm at **0.40765**, and the
projection is what got sealed.

That settles reading 1 of the three offered above, with evidence rather than by elimination.
**0.4063 was never a board figure.** It is the lab's estimate of one, it is accurate to 0.0013, and
the board's own answer has been sitting in the frozen table the whole time.

Publishing 0.40765 is therefore not raising our number. It is replacing a projection with the
measurement it was projecting, and the measurement is the board's.

**#19: a number that predicts a measurement must never be stored in the same shape as the
measurement.** 0.3884 and 0.4063 sit side by side in one sentence, one an observation and the
other a forecast, and nothing in the sentence says which is which.
