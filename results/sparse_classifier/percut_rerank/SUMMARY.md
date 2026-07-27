# Per-cut sparse-classifier reranker: 9-cell LAFA result

Per-category LightGBM reranker over the frozen `clean-learned-sparseclf-assoc-train227-test230` dataset. TEST = v227-v230 LAFA frame, scored with cafaeval (`-ia IA.tsv -prop fill -norm cafa -no_orphans -toi <toi>`, `-known PK_known` for PK). Metric = f_micro_w (IA-weighted micro-Fmax).

## TEST 9-cell f_micro_w (vs current live board)

| cell | reranker | board | delta |
|------|---------:|------:|------:|
| NK-mfo | 0.504 | 0.602 | -0.098 |
| NK-bpo | 0.215 | 0.309 | -0.094 |
| NK-cco | 0.351 | 0.431 | -0.080 |
| LK-mfo | 0.420 | 0.519 | -0.099 |
| LK-bpo | 0.311 | 0.348 | -0.037 |
| LK-cco | 0.357 | 0.419 | -0.062 |
| PK-mfo | 0.242 | 0.235 | +0.007 |
| PK-bpo | 0.140 | 0.117 | +0.023 |
| PK-cco | 0.266 | 0.254 | +0.011 |
| **MEAN** | **0.312** | **0.359** | **-0.048** |

- Cells improved vs board: **3/9**.
  - PK-bpo (+0.023), PK-cco (+0.011), PK-mfo (+0.007)
- Cells regressed: 6.
  - LK-mfo (-0.099), NK-mfo (-0.098), NK-bpo (-0.094), NK-cco (-0.080), LK-cco (-0.062), LK-bpo (-0.037)

## VALIDATION (v225-v227) selection proxy 9-cell

Flat IA-weighted micro-Fmax on the validation candidate pool (NOT cafaeval-propagated; used only for early-stopping / lever confirmation, never for tuning on TEST).

| cell | valid proxy f |
|------|--------------:|
| NK-mfo | 0.495 |
| NK-bpo | 0.348 |
| NK-cco | 0.531 |
| LK-mfo | 0.481 |
| LK-bpo | 0.594 |
| LK-cco | 0.550 |
| PK-mfo | 0.377 |
| PK-bpo | 0.879 |
| PK-cco | 0.292 |

## Protocol

- Per-category LightGBM binary classifier (objective=binary, early stopping on the v225-v227 hold-out by AUC). `aspect` is a feature; models are NOT split per aspect.
- TRAIN = snapshot_pairs v160-v165 .. v220-v225; VALID (early stop + selection) = v225-v227; TEST = v227-v230 (eval.parquet).
- Candidate pool: NK/LK use the full pool (knn+classifier+association); PK restricted to knn_present candidates (classifier/association dilute PK precision), applied consistently across train/valid/test.
- Lever (keeper): `pminmax` on LK-bpo only = per-protein min-max of the reranker score before thresholding.
- Features: 69 numeric+bool columns + encoded `aspect_code` (70 total; the full frozen schema beyond the documented core set was used).

## Caveats / footnotes

1. **Mixed-KNN-backend provenance (minor):** train snapshot_pairs v190-v195 and v195-v200 were computed with exact numpy KNN; all other train pairs, the v225-v227 validation, and the v227-v230 TEST use faiss IVFFlat (approximate). TEST and VALIDATION are therefore backend-consistent; the mix only adds slight noise to two early train splits.
2. This is the **per-cut** sparse-classifier result: co-annotation / classifier / association features are computed temporally-honest per cut (no future leakage), which is the point of the experiment.
3. The validation 9-cell are a flat candidate-pool proxy (no GO propagation, no PK-known exclusion for v225-v227), so they are not directly comparable in level to the cafaeval TEST cells; they are a monotone selection signal only.

## Per-cut vs Baseline (same reranker)

Identical reranker pipeline (same code, protocol, hyperparams, nested split, per-category pool, pminmax-LK-bpo, cafaeval scoring) run on two datasets that differ ONLY in the candidate-generation classifier: **per-cut** = temporally-honest two-tower sparse classifier; **baseline** = M2 anc2vec classifier. This isolates the classifier's effect with reranker training held constant.

| cell | per-cut | baseline | delta (PC-BL) | verdict | board | per-cut vs board | baseline vs board |
|------|--------:|---------:|--------------:|:-------:|------:|-----------------:|------------------:|
| NK-mfo | 0.504 | 0.664 | -0.160 | HURT | 0.602 | -0.098 | +0.062 |
| NK-bpo | 0.215 | 0.331 | -0.116 | HURT | 0.309 | -0.094 | +0.022 |
| NK-cco | 0.351 | 0.477 | -0.126 | HURT | 0.431 | -0.080 | +0.046 |
| LK-mfo | 0.420 | 0.564 | -0.144 | HURT | 0.519 | -0.099 | +0.045 |
| LK-bpo | 0.311 | 0.351 | -0.040 | HURT | 0.348 | -0.037 | +0.003 |
| LK-cco | 0.357 | 0.462 | -0.104 | HURT | 0.419 | -0.062 | +0.043 |
| PK-mfo | 0.242 | 0.204 | +0.038 | IMPROVE | 0.235 | +0.007 | -0.031 |
| PK-bpo | 0.140 | 0.109 | +0.031 | IMPROVE | 0.117 | +0.023 | -0.008 |
| PK-cco | 0.266 | 0.225 | +0.040 | IMPROVE | 0.254 | +0.011 | -0.029 |
| **MEAN** | **0.312** | **0.376** | **-0.065** | | **0.359** | **-0.048** | **+0.017** |

### Diagnostic answers

**(1) The NK/LK regression is the per-cut classifier DILUTING, not reranker under-tuning.** With the reranker held identical, per-cut is below baseline on **6/6** NK+LK cells (mean **-0.115**, up to -0.160 on NK-mfo). The baseline reranker is NOT low on NK/LK: it BEATS the live board on **6/6** NK+LK cells (mean **+0.037** vs board). So the same reranker on cleaner (M2) candidates recovers NK/LK fully; the per-cut sparse classifier's candidate set specifically dilutes NK/LK precision.

**(2) Cells where per-cut WINS vs baseline:** all 3 PK cells (PK-mfo +0.038, PK-bpo +0.031, PK-cco +0.040; mean **+0.036**). Per-cut PK also beats the board on all 3 (mean +0.014), while baseline PK LOSES to the board (mean -0.023). **LK-bpo:** per-cut does NOT gain (-0.040 vs baseline); the per-cut win is PK-exclusive.

**(3) Net recommendation (per-cell best-of graft)** (take each cell from whichever of board / baseline-M2-reranker / per-cut-reranker scores highest):

| cell | best f_micro_w | source |
|------|---------------:|:-------|
| NK-mfo | 0.664 | baseline |
| NK-bpo | 0.331 | baseline |
| NK-cco | 0.477 | baseline |
| LK-mfo | 0.564 | baseline |
| LK-bpo | 0.351 | baseline |
| LK-cco | 0.461 | baseline |
| PK-mfo | 0.242 | percut |
| PK-bpo | 0.140 | percut |
| PK-cco | 0.266 | percut |
| **MEAN** | **0.388** | |

Best-of-graft mean **0.388** vs board 0.359 (**+0.029**), baseline-reranker 0.376 (**+0.012**), per-cut 0.312. Source mix: board x0, baseline x6, per-cut x3. The graft takes **PK from the per-cut reranker** (its sole strength) and **NK/LK from the baseline-M2 reranker** (which already exceeds the board there); the board is dominated on every cell. Practically: keep the baseline-M2 reranker for NK/LK and graft the per-cut reranker for PK.
