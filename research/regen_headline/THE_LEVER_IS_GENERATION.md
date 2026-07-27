# The +0.02245 lever is genuine generation, not rescored inheritance

`storage/cooc_experiment/lever_mechanism_partition.py` (+ `.json`, `.log`)

## The doubt (the author's, and mine)

The rungs-in-IA run (`generator_rungs_ia.json`) showed the full-GO classifier's true extras carry
almost no MARGINAL IA: the pool's closure already covers 2.1x the IA weight we submit (129 terms per
protein propagate to 358). If the true extras add no new IA, where does the lever's +0.02245 in
f_micro_w come from?

The hypothesis, from `score_the_extras.py:355`: it keeps an extra when `g not in pl`, where `pl` is
the pool's SUBMITTED terms, not its closure. So an extra can be an ANCESTOR of a submitted term, a
cell `prop=fill` was already filling by inheritance. Submitting it explicitly with S's score REPLACES
that inherited value. So the lever might be RESCORING what fill already gave us, with a scorer better
than max-over-descendants. If so it would be much bigger than one cell: a claim about fill's
inheritance being improvable everywhere, on all nine cells.

## The test

Arm B's frozen submission (`lever_submission_top50.tsv`: 39,808 pool rows + 38,173 tagged extras).
Partition the extras by whether each sits inside the pool's propagated closure, then evaluate four
submissions on the board's line, same harness, to the digit.

| submission | rows | f_micro_w | delta |
|---|---|---|---|
| A pool only | 39,808 | 0.22282 | anchor (0.22282) |
| B + all extras | +38,173 | 0.24527 | +0.02245 (sealed) |
| A + IN (rescored inheritance) | +6,028 | 0.22359 | **+0.00077** |
| A + OUT (new candidates) | +32,145 | 0.24073 | **+0.01791** |

Both preconditions hold to the digit.

## The verdict

**GENERATION lever: new candidates carry it.** Rescoring the inheritance is worth +0.00077, noise.
The +0.01791 comes from terms genuinely OUTSIDE the pool's closure, things fill never gave us in any
form. The "rescore fill's inheritance across all nine cells" direction is dead: there is no signal
there.

## Why this is consistent with the low IA-marginal precision

The 32,145 OUT candidates have low raw IA precision (~2%). What makes them pay is S: it submits the
better half by its own score, concentrating the true, IA-bearing OUT candidates out of the mass of
false ones. That is exactly why arm C (raw sigmoid over the same candidates) LOSES at top50 and arm B
(S) wins by +0.0225. The lever was always generation-plus-a-scorer-that-can-choose; this closes the
mechanism.

## Standing caveats, unchanged

The lever is a LAB delta (+0.0326 cumulative, guard -> prefilter -> +extras). It does NOT transfer to
the board frame (PK-BP +0.0326 lab vs -0.0762 board gap; we do not reproduce the board and the frames
do not mix). Its value is that it corrects ch6's "retrieval is not what binds": a generator at genuine
candidate generation, scored properly, adds real BP terms the pool never proposed. The generator's
representation and head are NOT the bottleneck (`generator_rungs_ia.json`: learned codes -0.0065,
BP-only head -0.0019 at matched volume, both in IA).
