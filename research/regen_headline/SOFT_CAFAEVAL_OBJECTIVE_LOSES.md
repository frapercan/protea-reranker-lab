# A soft IA-weighted-F objective loses to lambdarank, in every frame

`storage/cooc_experiment/soft_cafaeval_objective.py` (+ `.json`, `.log`, `soft_cafaeval_predictions.npz`)

## The question (the author's)

Optimize a soft ontological metric, not a rank proxy. `fuse_listwise.py` proved "the objective was
the whole 0.045" (the neural listwise lost -0.0355 at equal input against the deployed lambdarank).
If the objective is the whole game, change the objective: keep LightGBM and the 72 reranker features,
swap ONLY the loss from lambdarank to a differentiable IA-weighted micro-F.

The soft objective, with `p_i = sigmoid(s_i)`, IA weight `w_i`, propagated label `y_i`, and
`F = 2*TP/(PM+AM)`, recomputing `(TP, PM, F)` each round and per-sample
`grad_i = -a_i p_i(1-p_i)`, `hess_i = |a_i| p_i(1-p_i) + eps`, `a_i = (2 w_i/(PM+AM))(y_i - F/2)`.

## One variable, and the precondition held

Same 72 features, same pk-bpo pool rows (16.5M train / 131k pos), same past/val split, same
leaves/lr/rounds. Arm A = lambdarank, arm B = the soft-F. **Precondition PASSED: arm A legacy =
0.22193, within 0.009 of the deployed 0.21269.** So the anchor is reproduced and the comparison is
valid.

## The result: it loses, decisively, in every frame

| frame | A lambdarank | B soft-F | B - A |
|---|---|---|---|
| legacy (lab, `max_terms=500`) | 0.22193 | 0.11570 | -0.10623 |
| vectorised (board, `max_terms=None`) | 0.13852 | 0.11570 | -0.02282 |

The fair frame is VECTORISED (the board's): there, non-positives are stored, so both arms submit
their whole output and the comparison is pure ranking/calibration quality. **B loses by 0.023**, 7x
the 0.0034 fold noise. The gate fails.

Why the legacy gap is larger and misleading: the soft-F emits `sigmoid` (all positive), so the legacy
parser (which DROPS non-positives) has nothing to drop and B is parser-invariant at 0.11570. Arm A
submits raw lambdarank scores WITH negatives, so legacy drops them as a free prefilter and lifts A to
0.22193. That prefilter is an artifact of the parser, not of the objective.

## Fairness check: B gets no rescue from the prefilter either

Giving B the same negative-prefilter freedom (submit its raw logits, let legacy drop the negatives):
raw-B legacy = 0.11114, raw-B vectorised = 0.10982. **B stays ~0.111 under every scoring choice**,
nowhere near A. B's ranking is genuinely worse than lambdarank's; the scale/parser choice does not
save it.

## What this closes

Both routes to beating the deployed lambdarank on the scalars are now measured negatives:
- **objective** (this): soft-F loses -0.023 (board frame).
- **architecture** (`fuse_listwise`): neural listwise loses -0.0355 at equal input.

lambdarank's per-protein ranking objective aligns with what f_micro_w rewards better than a global
soft-F does. Combined with the rankpct finding ("every technique lever is negative/flat: the deployed
recipe was already the best") and the generator rungs (representation and head both negative in IA),
the weight of evidence is unambiguous: **the deployed reranker is near-optimal; the only channel with
signal is GENERATION** (the classifier-extras lever, +0.02245, which adds NEW candidates outside the
pool's closure). The BP wall is a generation problem, not a scoring one.

## Caveat, stated

This shows THIS soft-F implementation (global micro-F at the tau=0 operating point, Gauss-Newton hess)
loses, not that no ontological objective can win. But it is the natural one-variable form of the
author's proposal, and it agrees with fuse_listwise. The end2end-over-vectors route does not earn its
-0.0355 handicap back, because its precondition (win on scalars first) failed.
