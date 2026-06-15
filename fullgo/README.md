# fullgo: how PROTEA reached first place on the LAFA benchmark

This folder holds the system that took PROTEA to **first place** on the LAFA continuously
updated public benchmark for protein function prediction. This README is the story of how it
works and how it got there, in plain terms. The exact numbers, recipe, and step-by-step
commands live in `RESULTS.md` and `REPRODUCE.md`.

## The result

On the benchmark's sealed September 2025 to March 2026 target frame (7,401 proteins), scored
with the benchmark's own harness, PROTEA leads the public leaderboard:

| Method | No knowledge | Limited knowledge | Partial knowledge | Overall |
|---|---|---|---|---|
| **PROTEA (this work)** | **0.477** | **0.482** | **0.215** | **0.391** |
| TransFew (previous leader) | 0.428 | 0.485 | 0.230 | 0.381 |
| FunBind | 0.441 | 0.451 | 0.205 | 0.366 |
| PROTEA, similarity search only | 0.412 | 0.394 | 0.165 | 0.324 |

The score is the benchmark's IA-weighted micro-F, averaged over the three branches of the Gene
Ontology, reported for the three standard settings: proteins about which nothing was previously
known, proteins with limited prior knowledge, and proteins that were already partly characterised.

## The story

**Where we started.** PROTEA predicts a protein's functions by finding the most similar proteins
whose functions are already known and borrowing their annotations. This similarity search is strong
when nothing else is known about a protein, and on its own it placed PROTEA level with the best
methods that use no machine learning. But it left a gap to the leaders, and that gap sat entirely in
the cases where a protein is already partly understood.

**What was missing.** Similarity search can only ever propose functions that some neighbour already
carries, and it ignores what a protein is already known to do. Closing the gap meant adding two
things: a predictor that is not limited to the neighbours, and a way to use prior knowledge about a
protein.

**The three ideas we added**, with a final stage that learns how much to trust each one:

1. **A second, complementary predictor.** A model reads each protein's learned representation
   directly and proposes functions across the entire catalogue at once, so the system is no longer
   capped by what its neighbours happened to carry.

2. **Averaging away luck.** That predictor is trained several times from different starting points
   and its proposals are pooled, so a single unlucky run can no longer drive a prediction. This
   steadies the rarer, harder calls in particular.

3. **Using what a protein is already known to do.** For a protein that is already partly
   characterised, the system asks which functions habitually accompany the ones it is already known
   to perform, paying special attention to links that cross from one branch of the ontology to
   another. Those cross-branch links carry information that the protein's sequence alone does not.

A final combiner weighs the similarity search, the direct predictor, the protein's own weaker prior
annotations, and the association signal, and learns when to trust each one. The association signal is,
by design, silent for proteins about which nothing is known, so it cannot invent knowledge where there
is none.

**Why the result is trustworthy.** Every choice was made on an earlier period of data and the system
was then measured a single time on the later, sealed period. As an extra guard, the earlier period was
itself split into a part used for fitting and a part held back, and re-deciding the ingredients on that
held-back part gives the same verdict: each idea genuinely helps. The improvements on the held-back
data are real but smaller than on the final frame, so the honest reading is a clear first place driven
by ingredients that are independently confirmed to help, rather than a headline margin reported to the
last digit.

## Where the gap remains

PROTEA now leads by a wide margin where nothing is known about a protein, draws level with the best
method where knowledge is limited, and trails only on the hardest cases: the rare, newly assigned
functions of partly known proteins. That is where the remaining headroom is, and it is the natural
target for future work (training the direct predictor at larger scale for rare functions).

## What is in this folder

- `RESULTS.md`: the full trajectory, every number, what was tried and rejected, and the
  leakage-validation table.
- `REPRODUCE.md`: the exact commands and the definitions of the two data periods.
- `train_classifier_m2.py`, `seed_average.py`, `assoc_feature.py`, `ensemble_seal.py`,
  `select_cv.py`: the scripts behind the three ideas, the combiner, and the held-out check.
- `config.yaml`: configuration (model ids, data periods, paths).

The whole system uses frozen, pre-computed protein representations and fits on a single 12 GB GPU.
Everything is held to one rule: any change must prove itself on the earlier period before it is
measured on the sealed one.
