# Per-category LightGBM reranker vs raw-KNN champion - TEST frame 227->230

Deployment-realistic test on the frozen `clean-learned-train227-test230` dataset 
(learned champion embedding `d8979601`, schema_sha `775611822dd9`). Metric: 
`f_micro_w` via the canonical cafaeval harness (prop=fill, norm=cafa, no_orphans, 
max_terms=500, th_step=0.001; IA/OBO v227). Reranker = per-category LightGBM 
lambdarank (one model per NK/LK/PK), 64 real features (association_* and 
classifier_* excluded as zero-filled), grouped by (snapshot_pair, protein, aspect), 
trained on v160..v227 with v225-v227 held out for early stopping.

## Pooled-per-aspect (one threshold/ns = the LAFA deployment number)

| aspect | champion | reranked | gate | d_reranked | d_gate |
|---|---|---|---|---|---|
| mfo | 0.3570 | 0.5185 | 0.3170 | +0.1615 | -0.0400 |
| bpo | 0.1602 | 0.3626 | 0.1418 | +0.2024 | -0.0184 |
| cco | 0.3105 | 0.4949 | 0.3076 | +0.1844 | -0.0029 |
| **MEAN** | **0.2759** | **0.4587** | **0.2555** | **+0.1828** | **-0.0204** |

## Per-cell (diagnostic, category-specific threshold)

| cell | champion | reranked | gate | d_reranked | d_gate |
|---|---|---|---|---|---|
| nk-mfo | 0.5549 | 0.6367 | 0.6367 | +0.0818 | +0.0818 |
| nk-bpo | 0.3619 | 0.4607 | 0.4607 | +0.0988 | +0.0988 |
| nk-cco | 0.4281 | 0.5488 | 0.5488 | +0.1207 | +0.1207 |
| lk-mfo | 0.4416 | 0.5654 | 0.5654 | +0.1238 | +0.1238 |
| lk-bpo | 0.3764 | 0.4870 | 0.4870 | +0.1106 | +0.1106 |
| lk-cco | 0.3905 | 0.5044 | 0.5044 | +0.1139 | +0.1139 |
| pk-mfo | 0.3017 | 0.4707 | 0.3017 | +0.1690 | +0.0000 |
| pk-bpo | 0.1401 | 0.3433 | 0.1401 | +0.2032 | +0.0000 |
| pk-cco | 0.2925 | 0.4886 | 0.2925 | +0.1961 | +0.0000 |
| **MEAN** | **0.3653** | **0.5006** | **0.4375** | **+0.1353** | **+0.0722** |

## Verdict

**Submit the FULL RERANKED config** (per-category LightGBM score for ALL categories). 
It maximizes the deployment metric: mean pooled-per-aspect f_micro_w 
**0.4587 vs champion 0.2759 (+0.1828)**, 
and also wins per-cell mean (0.5006 vs 0.3653).

The reranker lifts EVERY cell, including PK (pk-bpo +0.20, pk-cco +0.20, pk-mfo +0.17), 
contradicting the prior-frame expectation that PK regresses. The ADR-D43 GATE 
(reranker NK/LK + raw-KNN PK) is the WORST option on the pooled metric 
(0.2555, below champion): mixing reranker scores with raw-KNN 1-distance 
puts two incompatible score scales under one pooled threshold per namespace. 
Per-cell the GATE looks fine only because each cell self-thresholds; the deployment 
number is pooled, so the gate is rejected.

Caveats: train and eval are temporally separated (eval pair v227-v230 is future-only); 
no label is used as a feature; association_*/classifier_* zero columns excluded. 
The large lift warrants a one-shot leakage re-check before the LAFA submission, 
but the split and exclusions are clean.
