# The candidates the pool never asks for, scored by a model that can judge them

Receipt. Scripts `storage/cooc_experiment/{fullgo_ceiling,score_the_extras,bootstrap_the_lever}.py`
and `knn_vs_classifier_generator.py`, data in the matching `.json`, submission in
`lever_submission_top50.tsv`.

## The result

| | f_micro_w |
|---|---|
| A, the pool at the deployed threshold | **0.22282** |
| B, plus the classifier's top-50 extras scored by S | **0.24527** |
| **B - A** | **+0.02245** |
| **bootstrap 95% interval, 20 paired resamples** | **[0.01791, 0.02899]** |

**Twenty paired resamples, twenty positive.** The interval's lower bound is five times the 0.0034
fold-noise floor established for this cell. The precondition held: arm A returns 0.22282 against the
prefiltered anchor's 0.22288, an independent reproduction of PKW.8 to four decimals.

## Why it works, and why nobody saw it

The bar for a candidate expansion, from the two this campaign already measured: the co-occurrence
expansion the thesis reports adds candidates that are true **1.85%** of the time and converts to
+0.002. Descending the GO DAG from our own predictions adds them at **0.19%** and collapses the real
arm from 0.2131 to 0.0218. **The full-GO classifier proposes at 11.59%** (top5) and **9.84%**
(top20): six times the best we had, sixty times the structural route.

But the generator alone is not the lever. At top50 the **same 76,348 candidates** lose 0.0002 when
submitted with the classifier's own sigmoid and gain 0.0225 when scored by S. **The difference is
the scoring and nothing else.**

The shape says the same thing. B rises and falls (+0.0064, +0.0197, **+0.0225**, +0.0141, +0.0094 at
top 5/20/50/100/200); C rises and falls earlier and lower (+0.0051, +0.0114, -0.0002, -0.0041,
-0.0028). That is `prop=fill`'s tax: every submitted false extra forfeits its ancestor's free
inheritance, so **a better score does not remove the tax, it moves the depth at which the tax catches
you**, from top20 to top50.

## What S is, and the constraint that defines it

The extras carry **no kNN features**: the neighbour path never ran over candidates the classifier
proposed, which is the coverage hole PKW.1 found. So S uses only what is defined for **any**
(protein, term) pair: the full-GO logit, the `d8979601` protein code (2048-d), the two-tower GO
sparse code (1024-d), anc2vec (200-d), the term's information accretion, its parent count. It is
trained on the **pool's** labelled rows of v160..v225, early-stopped on v225-v227, and **transferred**
to the extras. It never learns to lean on a feature that will be absent when it has to judge.

## What this corrects in the thesis

Chapter 6 says retrieval is not what binds, and rests that on the co-occurrence expansion's +0.002.
**With a generator at 11.59% and a scorer that can use it, adding candidates pays +0.0225 with an
interval that excludes zero.** The same experiment explains why the earlier reading was reasonable:
with the raw sigmoid those same candidates lose. It was never that the pool sufficed. It was that we
had nothing that could score what lay outside it.

## What it does not do, stated plainly

**It does not flip PK-BP.** Cumulatively, in the lab frame: guard_only 0.21269, prefilter 0.22282
(+0.0092), extras scored 0.24527 (+0.0225), so **+0.0326** in total. The board gap is **-0.0762**
against TransFew's 0.2943, and **the +0.0326 is a lab delta that must not be subtracted from a board
gap**: we cannot reproduce the board (`WE_DO_NOT_REPRODUCE_THE_BOARD.md`) and the frames do not mix.

**The kNN remains the better generator.** At matched budget it reaches 72.0% of the ground-truth
weight against the classifier's 62.1%, and only **1,817** IA of truth is classifier-only against
**47,442** both reach. This is not a replacement for the pool. It is a small, scorable complement,
and the complement is what pays.

**It is a floor.** The generator is `classifier_6plm_asl`: 2026-06-14, six raw PLMs rather than the
learned k-WTA champion, no provenance beyond `mu`/`sd`/`vocab`, architecture recovered from tensor
shapes. The author called it old and unoptimised before it was run, and that is precisely why the
number matters: **this is what the worst version of the idea is worth.**
