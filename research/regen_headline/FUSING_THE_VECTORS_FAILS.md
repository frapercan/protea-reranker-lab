# Against the system we actually run, fusing the vectors does not help and the architecture loses

Receipt. Script `storage/cooc_experiment/fuse_listwise.py`, data `fuse_listwise.json`.

## The precondition finally passed, and that is what makes this readable

**Arm A, LightGBM lambdarank over the 72 scalars, scores 0.21969 against the deployed recipe's
0.21269**: within 0.007, so for the first time in this line of work the baseline is the system we
run rather than a stand-in.

That single fact explains the 0.045 that every earlier run in this file's lineage had to caveat.
The same model with a **binary** objective scored 0.16791. **The objective was the whole gap.**

## The result

Train v160..v225, early-stop on v225-v227, score v227-v230 blind. Three arms, because "fuse" hides
two variables.

| arm | what it is | f_micro_w | vs A |
|---|---|---|---|
| **A** | LightGBM lambdarank, 72 scalars (**the deployed shape**) | **0.21969** | |
| **B** | neural listwise, the same 72 scalars | 0.18415 | **-0.03554** |
| **C** | neural listwise, scalars **and** the raw vectors | 0.18595 | **-0.03374** |
| **C - B** | what the vectors add on top of the scalars | **+0.00180** | bootstrap 95% CI [0.00024, 0.00363] |

**The gate fails on both of its conditions.** `C - B` is +0.0018, under the 0.0034 noise floor for
this cell, and C loses to A by 0.034. **`VECTORS_ADD: False`.**

## What it says

**The architecture loses at equal input.** B minus A is **-0.0355**: given exactly the same 72
scalars, the cross-attention model is worse than the gradient-boosted tree. The problem was never
that the reranker could not fuse; it fuses better than the thing we replaced it with.

**The vectors add nothing measurable.** +0.0018 with a CI that clears zero but not the noise floor.
Given a model that already holds every scalar, the raw representations carry no usable extra signal
for this cell.

**And it retires the last three results in this line.** The joint model's +0.07526 and +0.02053 were
measured against a binary-objective GBDT at 0.167. Against the real lambdarank at 0.2197 the same
model loses by 0.034. Those numbers were never wrong; **the baseline was**, and a win over a
stand-in is not a win.

## Discipline #1, demonstrated in the logs of its own experiment

Arm C reached a validation **AUPRC of 0.4381**, far above anything else in the run, and scored
**f_micro_w 0.18595** against arm A's 0.21969. The ranking measure ordered the arms opposite to the
metric that decides, for the fifth time in this campaign. **f_micro_w decides, never AUC**, and this
run is the cleanest demonstration of why: the model that ranks best within each protein is not the
model that scores best under one global threshold, which is precisely the mismatch chapter 6
describes.

## What survives

The withholding levers, which are gated, cheap, and measured against the deployed recipe rather
than against a stand-in: prefilter the reranker **+0.0092** (sd 0.0034, unanimous across ten protein
folds) and the classifier arm **+0.0084** (sd 0.0058, nine of ten). They remain the only things this
campaign has found that move PK-BP without losing to the system already running.

The signal store keeps its case, but not this one. The export it stands in for took 109 seconds and
delivered 100% coverage, which is worth having on its own terms; what it does **not** buy, on this
evidence, is a joint model that beats the reranker.

**#21: a baseline you trained is not the system you run. Reproduce the deployed number first, or
every delta above it is measured against your own weaker work.**
