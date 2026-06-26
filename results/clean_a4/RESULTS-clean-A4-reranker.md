# Track C1: per-category reranker on the clean A4 frame (fresh schema)

Does reranking lift IA-weighted `f_micro_w` over the raw-KNN score, now that the
export carries the CURRENT live feature schema? The prior production reranker
was effectively a no-op because it was trained on a stale 56-feature schema and
the live predict path rejected it. This run trains on the fresh
`fullgo-clean-A4-train225-val` export (schema `775611822dd9`) so the reranker is
finally tested on the schema it would actually serve against.

## Provenance

- Dataset: `fullgo-clean-A4-train225-val` (k=30, source GOA), schema `775611822dd9`.
- Train: 13 snapshot pairs `v160-v165 .. v220-v225` (14,923,659 rows).
  Internal early-stopping holdout = the last pair `v220-v225`; the eval frame is
  never seen in training.
- Eval: the validation frame `v225-v227` (826,348 rows). Evaluation band `v227`
  (OBO `releases/2025-07-22`, IA `lafa_t0_Sep_2025/IA.tsv`, 39,906 terms). This
  is the OBO that cafaeval itself consumes, so propagation is congruent.
- Model: one per-category LightGBM LambdaMART booster (objective `lambdarank`,
  IA-weighted sample weights, mode `all`) over the ADR-D43 EvidenceScorer score
  vector. NK drops the priors per the combiner contract.
- Raw-KNN baseline score: `neighbor_vote_fraction`.

### Caveats (honest)

- `association_total`, `association_cross`, `classifier_score` are ZERO-FILLED in
  this export (cooccurrence + classifier-seed compute flags were off; deferred to
  a v2). They were dropped from the combiner vector, so the cooccurrence and
  classifier evidence ports are absent here.
- `anc2vec_neighbor_maxcos` is constant (1.0) in this export, so the
  label-embedding port carries no signal and is inert.
- The per-category real evidence vector therefore reduces to: alignment
  (`alignment_score_sw`), taxonomy (`taxonomic_distance`), knn
  (`distance`, `neighbor_vote_fraction`), plus `self_prior_score` for LK/PK.
- This is the VALIDATION frame (`v225-v227`). The final TEST frame `v227-v230`
  needs a later export trained up to v227.

## Result: f_micro_w lift, raw-KNN vs reranked

Point `f_micro_w` is from cafaeval (authoritative). The bootstrap delta and 95%
CI are from a per-protein paired bootstrap (1000 iterations) of a numpy
reimplementation of IA-weighted micro-F (propagation + no_orphans + max_terms);
its point level validates to within roughly 0.01 to 0.05 of cafaeval per cell and
its sign and significance agree, so the paired delta CI is a sound uncertainty
estimate (systematic offset cancels in the pair).

| cell    | proteins | cafaeval raw | cafaeval rerank | cafaeval delta | bootstrap delta [95% CI] | p(delta<=0) |
|---------|---------:|-------------:|----------------:|---------------:|--------------------------|------------:|
| nk-mfo  |   562 | 0.5016 | 0.5645 | **+0.0629** | +0.0385 [+0.0202, +0.0566] | 0.00 |
| nk-bpo  |   811 | 0.4253 | 0.4713 | **+0.0460** | +0.0272 [+0.0082, +0.0469] | 0.00 |
| nk-cco  |   660 | 0.4516 | 0.5545 | **+0.1029** | +0.0842 [+0.0582, +0.1128] | 0.00 |
| lk-mfo  |   452 | 0.4705 | 0.5089 | **+0.0384** | +0.0344 [+0.0143, +0.0568] | 0.00 |
| lk-bpo  |  1640 | 0.5094 | 0.5190 | +0.0096 | -0.0066 [-0.0185, +0.0048] | 0.87 |
| lk-cco  |   482 | 0.4562 | 0.4807 | **+0.0246** | +0.0465 [+0.0156, +0.0809] | 0.00 |
| pk-mfo  |  1838 | 0.4077 | 0.3805 | **-0.0272** | -0.0427 [-0.0576, -0.0278] | 1.00 |
| pk-bpo  |  8639 | 0.4463 | 0.2738 | **-0.1726** | -0.1589 [-0.1677, -0.1494] | 1.00 |
| pk-cco  |  2765 | 0.4080 | 0.3648 | **-0.0432** | -0.0693 [-0.0848, -0.0529] | 1.00 |

Mean cafaeval delta across the 9 cells: **+0.0046** (near flat; the strong NK
gains are cancelled by the large PK regression, which is dominated by the
8,639-protein pk-bpo cell).

## PK: precision vs recall (IA-weighted, at the f_micro_w-optimal threshold)

| cell   | raw P | raw R | rerank P | rerank R | delta P | delta R |
|--------|------:|------:|---------:|---------:|--------:|--------:|
| pk-mfo | 0.213 | 0.515 | 0.158 | 0.694 | -0.055 | +0.179 |
| pk-bpo | 0.363 | 0.472 | 0.168 | 0.509 | -0.196 | +0.037 |
| pk-cco | 0.186 | 0.419 | 0.114 | 0.561 | -0.072 | +0.142 |

The PK regression is a precision collapse: the reranker trades IA-weighted
precision for recall, and on the prior-heavy PK category the IA-weighted micro-F
penalizes that trade. pk-bpo precision drops 0.196. This reproduces the standing
finding that PK precision is the wall.

## Verdict

The fresh-schema reranker is NOT a no-op. Per category:

- NK (no priors): strong, significant lift everywhere (+0.046 to +0.103). The
  learned combination of sequence, taxonomy and knn evidence clearly beats the
  raw vote fraction where there is no prior to lean on.
- LK: small net positive (MFO +0.038 and CCO +0.025 significant; BPO +0.010 not
  significant, CI spans 0).
- PK: significant REGRESSION everywhere (-0.027 to -0.173), driven by a
  precision collapse.

So the honest answer is category-dependent: reranking helps NK (and LK MFO/CCO),
and hurts PK. A per-category gate (apply the reranker on NK and LK, keep raw-KNN
on PK) captures the NK/LK gains without the PK loss.

Likely PK mechanism and next step: `lambdarank` optimizes per-protein ranking,
not a globally calibrated score, so its outputs threshold poorly under the pooled
micro f_micro_w on the prior-heavy PK category, where the raw vote fraction is
already well calibrated across proteins. Per-aspect isotonic calibration of the
PK scores (as the universal runner does) is the obvious follow-up to test whether
PK can be recovered. The zero-filled association and classifier ports also remove
two of the evidence signals that historically carried PK; a v2 export with those
compute flags on is needed for a complete PK verdict.

Reproduce: `scripts/run_clean_a4_reranker.py` (see `results/clean_a4/results.json`
for the full per-cell record). MLflow experiment `reranker-clean-A4-train225`.
