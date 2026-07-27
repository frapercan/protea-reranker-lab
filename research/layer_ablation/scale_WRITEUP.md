# The representation ablation at scale: what held, and what did not reproduce

Receipts: `scale_result.json` (`scale_train.py`) and `scale_hardneg_result.json`
(`scale_hardneg.py`), read against `crown_result.json`, `knn_confirm_results.json`, and the
arm/objective code in `encoder_ablation.py`. Every number below traces to a present receipt.
These are KNN-only retrieval-encoder mean9 f_micro_w numbers over the nine cells; they are
CANDIDATES and a different measurement from the sealed 0.4063 reranked pipeline, which is untouched.

## The result (mean over seeds 42, 43, 44; all arms scored into the SAME 15k reference)

| arm (ankh-base) | objective | mean9 f_micro_w |
|---|---|---|
| L48 (served base) | cosine-lin | 0.1422 |
| L10 (raw best layer) | cosine-lin | 0.1411 |
| L10-std (per-dim z-score of L10) | cosine-lin | 0.1576 |
| mix-learned (softmax over L10,L19,L48) | cosine-lin | 0.1578 |
| **L48 (controlled)** | **hard-neg** | **0.1447** |
| served champion d8979601 | hard-neg, 100k scoring ref | 0.2150 |

## Within the controlled ablation, one lever moves the number: z-scoring

Wilcoxon + Holm on per-(cell,protein) IA-Fmax, 8589 shared proteins:

- **z-score is the lever, not the layer.** L10-std beats L10-raw by +0.0185 (p_holm 3.2e-39) and beats
  L48 by +0.0167 (p_holm 6.8e-33). L10-raw vs L48 is null: -0.0018 (p_holm 0.58). Choosing a "better"
  layer does nothing; standardizing its dimensions is what moves the number. The crown's headline
  finding, confirmed at 100k with the starvation confound removed.
- **Learned multi-layer mixing does not separate** from L10-std: -0.0009 (p_holm 0.96). Its softmax
  mass is near-uniform, [0.326, 0.356, 0.318] over layers [10, 19, 48], a mild L19 peak, exactly what
  the crown saw. Learned mixing buys nothing over standardizing one good layer.

## Two things that were expected to be levers and are NOT (both null in this harness)

- **Training pool size (15k -> 100k): null.** cosine-lin L48 is 0.1425 at 15k (crown) and 0.1422 at
  100k (scale). 6.7x more training proteins does not move it. The crown arms were not starved.
- **The hard-neg objective: null.** The controlled hard-neg L48 at 100k, trained by the SAME
  `_train_encoder(objective="hard-neg")` the champion's arm uses, scored into the same 15k reference,
  reaches only 0.1447, +0.0025 over cosine-lin. Mining embedding-near / GO-far pairs does not close the
  gap in this harness.

## The gap to the served encoder is RESOLVED: it is the BASE EMBEDDING

The re-trained arms plateau at 0.14 to 0.16; the served production encoder `d8979601` reaches 0.2150.
That gap is neither scale nor objective nor scoring reference. It is the base embedding.

The controlled proof (receipt `crown_control_apples_result.json`, 2026-07-11): the SAME
`_train_encoder(objective="hard-neg")`, the SAME 100k training pool, the SAME 15k scoring reference,
the SAME architecture (dict_dim 2048, top_k 128, seed 42), trained on the production-stored `08234f06`
ankh-base embeddings pulled from the database, reaches mean9 **0.22013**, reproducing the served
champion (0.2150, gap minus 0.005). The identical run on the local `scale_extract.py` layer-48
mean-pool (`scale_pool_emb/layer_48.npy`) reaches only **0.1447**. Only the base embedding differs, and
it is worth about 0.075 in mean9.

So the local layer-48 extraction is simply a much weaker retrieval base than the production `08234f06`
embedding. This also vindicates the earlier 0.2197 number (it is real, 0.220) while correcting its
interpretation: that control changed BOTH the pool source AND the embedding source, and the lever was
the embedding, not the pool size (a same-config hard-neg run on the local base at 100k is still 0.1447).

Two honest caveats. First, the exact extraction difference between the stored `08234f06` pipeline and
the local layer-48 mean-pool (pooling, normalization, or which tensor "last layer" maps to) is not yet
pinned; only that the difference exists and is large. Second, the ablation's arm findings (z-scoring is
the lever, etc.) were measured on the weaker local base; whether the z-score lever still helps on top
of the strong `08234f06` base is untested. The INTERNAL arm ordering stands (all arms shared the local
extraction and the 15k reference).

## What to carry forward (only the receipt-backed claims)

1. **z-scoring the base is the one real, significant retrieval lever** in this family (+0.0167 over L48,
   p<1e-32); it is the standardization, not the layer index, that matters.
2. **Null levers:** training pool size, the hard-neg objective, raw layer choice, and learned
   multi-layer mixing all fail to move the number.
3. **The base embedding is the lever between the arms and the champion**, proven: the same training on
   the production `08234f06` DB embeddings reaches 0.220 (champion), on the local layer-48 extraction
   only 0.1447. The local `scale_extract.py` mean-pool is a weaker base than production; pin the exact
   extraction difference next. The z-score lever was measured on the weaker base; retest it on `08234f06`.
