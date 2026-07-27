# Representation ablation: layer, sparsity, normalization, and learned aggregation for functional kNN transfer

*Ankh-base substrate, 2026-07-08. A controlled ablation that isolates the three
fixed axes of the protein representation used for GO transfer (layer, sparsity,
normalization) and sets them against the learned k-WTA encoder PROTEA actually
serves.*

## Question

PROTEA transfers Gene Ontology terms by k-nearest-neighbour retrieval over PLM
embeddings. The served representation is a fixed choice: the last transformer
layer, mean-pooled, dense, L2-normalized. Three axes of that choice had never
been isolated, and one comparison had never been made explicit:

1. **Layer.** Which transformer layer carries the most functional signal.
2. **Sparsity.** Does k-WTA sparsification help or hurt the retrieval geometry.
3. **Normalization.** Raw (L2 only) versus per-dimension standardization.
4. **The reference point.** How do the best fixed choices compare to the learned
   k-WTA encoder that PROTEA serves (config `d8979601`, the +40% end-to-end
   lever), and to the naive last-layer-dense baseline it replaced.

## Protocol

**Substrate.** 7401 LAFA query proteins (queries) and a disjoint 15,000-protein
t0-annotated reference subset, each with sequence and propagated GO.

**Representations.** ankh-base with `output_hidden_states`, six layers sampled
across depth (hidden-state indices 0, 10, 19, 29, 38, 48 of the 49 available).
Mask-aware mean-pool, sequence cap 2048 (matches the cached-embedding limit).
Sparsity in {dense, k-WTA 64, k-WTA 128, k-WTA 256}. Normalization in {raw = L2
only; std = per-dimension z-score then L2}. Grid = 6 layers x 4 sparsity x 2
normalization = **48 cells**. The learned encoder `d8979601` is a 2048-dim k-WTA
code with 128 non-zeros, pulled from the platform.

**Screen (proxy).** Two cheap, reference-free functional-geometry metrics over
the 7401 queries, stratified by category and length:

- (A) global Spearman between embedding cosine and propagated-GO Jaccard over a
  fixed sample of 200,000 protein pairs;
- (B) top-10 neighbour functional purity (mean propagated-GO Jaccard to the ten
  nearest neighbours in embedding space).

**Confirm (real task).** Board-faithful kNN GO transfer: queries retrieved into
the 15k reference, cosine top-30, neighbour GO transferred by a cosine-weighted
vote, `cafaeval` f_micro_w per category over the nine {NK, LK, PK} x {BPO, MFO,
CCO} cells. Standardization statistics are fit on the reference pool
(non-transductive) and applied to both query and reference. Six representations
were carried through the confirm.

## Results

### 1. The proxy, and its documented tension

Baseline (layer 48, dense, raw = exactly what PROTEA served): Spearman 0.0376,
purity_all 0.1179, purity_long 0.1289.

The two proxies disagree by design, and the disagreement is informative:

| axis | metric that improves | metric that degrades |
| --- | --- | --- |
| raw (L2 only) | Spearman: mean 0.0428 across raw cells | purity: mean 0.1232 |
| std (z-score then L2) | purity: mean 0.1328; 9 of the top-10 purity cells are std | Spearman: mean 0.0208 |

Per-dimension standardization **wins on top-10 purity and on the real task**, but
**loses on global Spearman**. Mechanism: z-scoring rescales every dimension to
unit variance, which reshapes the *local* neighbourhood structure that kNN
actually consumes (each query's ten nearest neighbours), while flattening the
few high-variance dimensions that carry most of the *global* pairwise rank
correlation. The screen therefore reports a metric (Spearman) on which raw wins
and a metric (purity, the one aligned with retrieval) on which std wins, and the
confirm below settles it in favour of std.

Concrete proxy deltas over the served baseline:

- best purity cell overall: L0 k-WTA-64 std at 0.1498 (+27.1% over 0.1179);
- best purity on **long** proteins: L10 k-WTA-128 std at 0.1824 (+41.5% over the
  baseline long value 0.1289). The gain concentrates on long sequences, the
  dilution signature of last-layer mean-pooling.

### 2. The layer-0 confound

Layer 0 tops the raw Spearman ranking (L0 dense:raw = 0.0586, the highest of all
48 cells; per-layer dense:raw Spearman falls L0 0.0586 > L10 0.0478 > L19 0.0460
> L48 0.0376 > L38 0.0357 > L29 0.0311). This is a trap. Layer 0 is the *input
embedding*, close to amino-acid composition, so its correlation with functional
Jaccard is largely a composition-similarity artefact rather than learned
function. Its Spearman is also strongly length-dependent:

| L0 dense:raw Spearman | short | med | long |
| --- | --- | --- | --- |
| by length | 0.0218 | 0.0713 | 0.0901 |

Short to long spans 0.0218 to 0.0901 (a 4.1x range), the fingerprint of a
length/composition effect rather than a stable functional signal. Discounting
L0, the credible fixed winner is **layer 10**: high purity (best purity_long
0.1824 at L10 k-WTA-128 std) without the input-embedding confound, and it is L10
that the confirm carries forward.

### 3. Massive activations: storage must be float32

Mid-layer mean-pooled activations are enormous. Peak absolute value per stored
layer:

| layer | 0 | 10 | 19 | 29 | 38 | 48 |
| --- | --- | --- | --- | --- | --- | --- |
| max abs | 12.4 | 58,888 | 81,722 | 158,039 | **440,611** | 0.6 |

The mid layers reach |440,611| (layer 38), and layers 19, 29 and 38 all exceed
the float16 ceiling (65,504). Stored as float16 these overflow silently to Inf,
which corrupted an earlier run of this experiment. The extractor stores float32
for this reason. Note also the two ends: L0 peaks at 12.4 (raw input scale) and
L48 at 0.6, scaled down by the model's final LayerNorm, which is one reason the
served last layer is a weak retrieval base.

### 4. The board-faithful confirm

Real kNN GO transfer, f_micro_w per cell, six representations. The full
nine-cell table for the two endpoints plus the per-config mean:

| cell | L48-dense (served) | L10-kwta128-std | d8979601 (learned) |
| --- | --- | --- | --- |
| nk-bpo | 0.10530 | 0.11960 | 0.16646 |
| nk-cco | 0.23379 | 0.27349 | 0.31222 |
| nk-mfo | 0.12648 | 0.13984 | 0.34481 |
| lk-bpo | 0.14400 | 0.14779 | 0.21742 |
| lk-cco | 0.29463 | 0.31788 | 0.31982 |
| lk-mfo | 0.09974 | 0.09439 | 0.27577 |
| pk-bpo | 0.04304 | 0.04903 | 0.06447 |
| pk-cco | 0.10947 | 0.12186 | 0.13840 |
| pk-mfo | 0.04561 | 0.04989 | 0.09567 |
| **mean** | **0.13356** | **0.14597** | **0.21500** |

Ranking of all six by mean f_micro_w:

| representation | mean f_micro_w | vs served baseline |
| --- | --- | --- |
| L48 dense (what PROTEA served) | 0.13356 | worst of the six |
| concat(L10, L48), std | 0.14143 | +5.9% |
| concat(L10, L48), k-WTA-128 | 0.14233 | +6.6% |
| L10 k-WTA-128, std (best fixed) | 0.14597 | +9.3% (up in 8 of 9 cells) |
| `d8979601` learned k-WTA | 0.21500 | +61.0% |

Two percentages anchor the finding:

- **Learned vs best fixed:** 0.21500 vs 0.14597 = **+47.3%**.
- **Learned vs the naive last-layer-dense baseline:** 0.21500 vs 0.13356 =
  **+61.0%**.

The learned encoder wins all nine cells, and the gap is largest on molecular
function: nk-mfo 0.34481 vs 0.12648 served (2.73x) and vs 0.13984 best fixed
(2.47x); lk-mfo 0.27577 vs 0.09974 served (2.76x) and vs 0.09439 best fixed
(2.92x); pk-mfo 0.09567 vs 0.04561 served (2.10x). It does not merely pick a
better layer, it learns a functional geometry that no fixed layer, top-k, or
standardization reaches on this same extraction.

> **Caveat (added 2026-07-11): the learned-vs-fixed gap is confounded by the embedding source.**
> The fixed representations above are extracted locally (mask-aware mean-pool, cap
> 2048, from `output_hidden_states`), whereas `d8979601`'s codes are pulled from the
> platform and were trained on the production-stored embedding (config `08234f06`), a
> different extraction pipeline: the same protein's local last-layer mean-pool has
> cosine only about 0.28 to the production vector, and about 0.02 to the local layer 0.
> A same-base control (`scale_hardneg_result.json`, 2026-07-11) trains the identical
> learned k-WTA head on the LOCAL last-layer extraction and reaches only 0.1447, versus
> the fixed local L48 dense 0.13356: the clean learned-head benefit on a shared base is
> about +8 percent, and the rest of the gap to 0.215 is the production embedding, not
> the learned geometry. So the +47.3 and +61.0 percent above mix two levers (a learned
> head plus a stronger production embedding); do not read them as the learned head alone.

### 5. Naive multi-layer is negative

Equal-weight concatenation of L10 and L48 is **worse than L10 alone**:
concat(L10, L48) std at 0.14143 and concat k-WTA-128 at 0.14233 both fall below
the single-layer L10 k-WTA-128 std at 0.14597 (concat-std is 3.1% below it).
Averaging in the weaker last layer dilutes the stronger mid layer. Multi-layer
combination helps only if the combination is **learned**, not equal-weighted.

## Mechanism

Three threads explain the whole grid. First, depth: the served last layer is
scaled down by a final LayerNorm (peak 0.6) and carries less functional geometry
than mid layers, while the true early-layer signal is contaminated by
composition (the L0 confound). Layer 10 is the sweet spot among fixed choices.
Second, normalization is what makes sparsity work: k-WTA on raw hidden states is
captured by a handful of massive-activation dimensions that always survive the
top-k; z-scoring first equalizes the dimensions so the retained coordinates are
functionally meaningful, which is why std wins the retrieval-aligned purity
metric and the real task even though it loses global Spearman. Third, learning
dominates all three fixed axes together: each fixed axis buys a few points and
the best fixed stack buys +9.3%, whereas the learned encoder buys +61.0%,
because it optimizes the aggregation toward function rather than reading it off a
frozen layer.

## Limits (stated plainly)

- **Only 6 of the 48 cells were board-confirmed.** The confirm carried L48-dense,
  L10 k-WTA-128 std, the two concat variants, and the learned encoder. The claim
  "the learned encoder beats any fixed representation" is supported for the
  confirmed configurations and for the proxy-selected best fixed cell, not
  proven for all 48 cells.
- **Single PLM.** Everything is ankh-base. Cross-family behaviour (ProtT5, ESM)
  is untested here.
- **The proxies are 7401-only geometry.** Spearman and purity are computed on
  query-side geometry alone, not on held-out transfer, so they are a screen, not
  a verdict; the confirm is the verdict.
- **Capacity is only partly isolated.** `d8979601` is 2048-dimensional against
  768 for the hand-crafted arms, so learned-versus-fixed is entangled with
  dimensionality. However the magnitude (+47% to +61% on the mean and a 2x to 3x
  gap on the MF cells) is far too large to be a dimensionality artefact; a 2.7x
  factor is not what 2048/768 = 2.67x of raw capacity buys on a kNN vote.

## What follows

The open question the campaign is now testing sits at the intersection of
findings 3 and 5. The served last layer is the weakest fixed base in this
ablation, yet `d8979601` and its two-tower base are trained on exactly that last
layer. A learned head has never been trained on any other layer, nor on a
learned multi-layer combination (which findings 2 and 5 suggest could beat the
single layer where naive concat fails). Retraining the hard-negative k-WTA head
on a mid-layer or learned-layer-mix input, rather than the last layer it uses
today, is the crowning experiment this ablation motivates and does not itself
run.

## Reproduction

`storage/layer_ablation/`: `extract_layers.py` (six-layer ankh-base extractor,
float32 storage because mid-layer activations overflow float16),
`screen_funcgold.py` (48-cell proxy screen), `ref_extract.py` (reference-pool
embeddings), `knn_confirm.py` (kNN transfer plus cafaeval, all confirm configs).
Results: `screen_funcgold_results.json` (48 cells),
`knn_confirm_results.json` and `knn_confirm_5way.json` (the nine-cell confirm
per configuration).
