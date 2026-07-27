# Text-as-generator does NOT beat the classifier: the apparent semantic payoff is a scale artifact

`score_the_extras_semantic.py`, `union_matched_volume.py`, `union_candidates_scored.npz`

## The arc, honestly

Phase 0.5 showed the semantic retrieval is real: a biomedical encoder ranks a protein's novel-missed
BP terms at AUC 0.78 (5.6x over random). The question was whether that retrieval CONVERTS into a
generation payoff in the cafaeval frame, the way the classifier-extras lever (+0.02245) does.

First reads suggested yes: `B_semantic` paid +0.01102 (per-protein top200), and at matched global
volume `semantic-only` beat `classifier-only` (+0.0122 vs +0.0067) at every budget. That looked like
an independent, stronger text generator (overlap with the classifier only 4.6%).

## The re-check that flipped it (standing norm: re-check suspiciously-clean results)

The matched-volume `semantic-only` delta was flat across V (0.01215 -> 0.01237) despite a low true
rate. Direct, scale-free candidate quality (marginal over the pool closure, which is what `prop=fill`
actually rewards/taxes), at matched V=113k:

| arm | NEW-true IA | false-NEW IA | true/false |
|---|---|---|---|
| classifier | **6,332** | 99,966 | 0.063 |
| semantic | 2,037 | 173,412 | 0.012 |

The classifier contributes 3x more new-true IA AND less false tax. By raw material it is the better
generator, so it MUST score higher at matched volume if cafaeval were measuring candidate quality. It
did not. Contradiction => the `semantic` cafaeval advantage is not candidate quality.

## The proof: fix the score, kill the scale

Submit each arm's top-V=113k extras at a SINGLE fixed score (0.42), removing the S-score scale as a
variable, so f_micro_w reflects only WHICH terms are in:

| arm | delta at fixed score |
|---|---|
| classifier | -0.00310 |
| semantic | -0.00044 |
| union | -0.00312 |

At a fair score NOBODY pays at this volume; the classifier even loses. The entire "semantic pays
+0.0122" was the S-score distribution interacting with cafaeval's threshold sweep (the documented
SCALE lever: cafa_eval depends on the score's scale). It was never candidate quality.

## What stands, corrected

- **Text-as-a-candidate-generator does NOT beat the classifier.** Scale-free, the classifier is 3x
  the generator the semantic channel is; at fair score the semantic bulk contribution is ~0.
- The Phase 0.5 retrieval AUC (0.78) is real but does NOT convert to a generation payoff. This is the
  protst_text pattern (signal exists, payoff ~+0.0016) reproduced at the generator level. A strong
  retrieval AUC is necessary, not sufficient; the fill tax is the wall.
- The classifier remains the only generation lever that pays, and only at small high-precision k
  (top20/top50), where the +0.02245 was measured; at V=113k even the classifier is net-negative.
- The semantic channel IS independent (4.6% overlap, ~1,755 true-new terms the classifier misses,
  worth ~2,037 IA), but that add is small and its false tax (173k IA) is brutal, so extracting it
  net-positive is not realistic as a bulk generator.

## Discipline note

I reported "+0.0122 semantic pays" to the author before this re-check. It was wrong: a scale
artifact. Caught by the marginal-IA contradiction + the fixed-score control. No lab cafaeval delta
between arms with different score distributions can be trusted without a fixed-score control; the
scale is a 0.088 lever and it masquerades as signal.
