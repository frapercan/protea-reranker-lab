# L10-std champion regeneration, Step 3 results (KNN-retrieval mean9 screen)

Author: Francisco Miguel Perez Canales. Date: 2026-07-12.

## Verdict (the #1 open question)

z-score-on-production (L10-std) does **NOT** beat the served L48 champion at the
KNN-retrieval level. Every L10-std arm lands **below** the champion by 0.026 to 0.033
mean9 f_micro_w. The debugging-era local-harness win (`scale_train.log`: L10-std vs L48
+0.0167) does **not survive** on production embeddings. This confirms the confound
flagged in `BETTER_KWTA_DESIGN.md`: the +0.0167 was an artefact of the local extraction
path, which does not reproduce production (local L48 0.1422 vs served champion 0.2150).

The champion baseline re-scored in this exact harness reproduces the pinned value to five
decimals (0.21501 vs pinned 0.21500), and the champion head applied to its production L48
base reproduces the pinned query codes (`query_d8979601.npy`) with sample mean abs diff
4e-7 (max 0.0325 = a handful of top-k boundary tie-flips, cosmetically irrelevant: the
mean9 is identical). So the harness and the apples comparison are trustworthy.

## Leaderboard (mean9 f_micro_w, KNN-retrieval only)

| rank | arm | dict | top_k | train pool | mean9 | vs champion |
|---|---|---|---|---|---|---|
| baseline | champion_L48_apples | 2048 | 128 | 100k (L48 raw) | **0.2150** | +0.0000 |
| 1 | L10std_d4096_k256_100k | 4096 | 256 | 100k | 0.1887 | -0.0263 |
| 2 | L10std_d2048_k128_100k (PRIMARY apples) | 2048 | 128 | 100k | 0.1878 | -0.0272 |
| 3 | L10std_d2048_k128_full | 2048 | 128 | 534k | 0.1863 | -0.0287 |
| 4 | L10std_d4096_k128_100k | 4096 | 128 | 100k | 0.1862 | -0.0288 |
| 5 | L10std_d2048_k256_100k | 2048 | 256 | 100k | 0.1853 | -0.0297 |
| 6 | L10std_d4096_k64_100k | 4096 | 64 | 100k | 0.1844 | -0.0306 |
| 7 | L10std_d2048_k64_100k | 2048 | 64 | 100k | 0.1823 | -0.0327 |

Winner (best L10-std) = **L10std_d4096_k256_100k = 0.1887**, but it beats the primary
apples arm by only +0.0009 (inside single-seed noise), and still trails the champion by
0.0263. The hyperparameter screen yields no material lever: dict_dim 2048 -> 4096 and
top_k 64 -> 256 move mean9 by at most ~0.006, all below the champion.

## Where the gap lives (per-cell, the stratification that matters)

| cell | champion (L48) | best L10-std (d4096/k256) | primary (d2048/k128) | full (534k) |
|---|---|---|---|---|
| nk-mfo | **0.3448** | 0.2585 | 0.2486 | 0.2372 |
| nk-bpo | 0.1665 | 0.1677 | 0.1671 | 0.1627 |
| nk-cco | 0.3122 | **0.3460** | 0.3390 | 0.3330 |
| lk-mfo | **0.2758** | 0.1507 | 0.1471 | 0.1613 |
| lk-bpo | **0.2173** | 0.1694 | 0.1779 | 0.1753 |
| lk-cco | 0.3200 | **0.3229** | 0.3207 | 0.3217 |
| pk-mfo | **0.0957** | 0.0732 | 0.0754 | 0.0753 |
| pk-bpo | **0.0644** | 0.0617 | 0.0608 | 0.0569 |
| pk-cco | 0.1384 | **0.1487** | 0.1538 | 0.1531 |
| mean9 | **0.2150** | 0.1887 | 0.1878 | 0.1863 |

Reading: the mean9 gap is an **MFO collapse**. Layer 10 loses molecular-function signal
badly (nk-mfo -0.086, lk-mfo -0.125, pk-mfo -0.022 vs the last layer). L10-std actually
**wins CCO** (nk-cco +0.034, lk-cco +0.003, pk-cco +0.010 vs champion) and roughly ties
BPO (nk-bpo +0.001, lk-bpo mixed). So the last layer carries MFO; the middle layer does
not. z-scoring a middle layer cannot recover what the layer choice discards. This is a
representation-level, not a geometry-level, finding.

## Does full-data or a different top_k/dict_dim help?

- **Full data: no.** Training the head on 534,060 v227-annotated proteins (vs 100k) gives
  0.1863, marginally **below** the 100k primary (0.1878). More training data does not help
  on the production base. This re-confirms the pool-size null (crown 15k -> 100k) now on
  the production embedding and with the KNN reference/eval held fixed, varying only the
  training pool. The head is not data-starved at 100k.
- **Hyperparameters: no material lever.** The 6-way screen (top_k in {64,128,256} x
  dict_dim in {2048,4096}) spans only 0.1823 to 0.1887. Bigger dict + bigger top_k trend
  slightly up but never approach the champion. top_k had never been swept before; it is now
  ruled out as the missing lever.

## Recipe + provenance (pinned, auditable)

- **Base**: Ankh-base layer 10, production `EmbeddingConfig 81436dba-1324-4536-bccc-122ac45dd9ba`
  (`layer_indices=[38]`, diff vs champion base `08234f06` = layer only). DB-stored values
  are L10/32 (uniform `embedding_scale=32` to fit halfvec fp16); read as-is, the per-dim
  z-score absorbs the /32 exactly. 615,960 accession-level embeddings materialised
  (527,861 distinct sequence-embeddings; some sequences map to multiple accessions).
- **z-score**: `mu = X.mean(0)`, `sigma = X.std(0) + 1e-6` per-dim, fit on the training
  pool (100k for the screen, 534k for the full arm). Saved as `head.scaler.npz` (`mu`,
  `sigma`, each 768-d), a same-stem sibling of `head.pt`.
- **Head (champion recipe verbatim)**: `Linear(768 -> dict_dim)` + `topk_real(top_k)` over
  `l2n(z-scored base)`; objective `hard-neg` = 300,000 random pairs (`sample_pairs`, seed 42)
  plus 2000 anchors x 30 mined embedding-near negatives; target `lin_pairwise` Lin GO-sim;
  Adam lr 1e-3; 150 epochs; bs 32768; float32; seed 42. Reuses
  `protea-reranker-lab/src/protea_reranker_lab/encoder_ablation.py` (`l2n`, `topk_real`,
  `sample_pairs`) and `sdr` (`GoDag`, `information_content`, `lin_pairwise`, `propagate`).
  Training uses a per-minibatch backward (gradient of the summed loss = sum of per-batch
  gradients, mathematically identical to the full-batch forward) so dict_dim 4096 and the
  534k pool fit the 12 GB card. Confirmed equivalent: the memory-safe re-run reproduced the
  full-forward numbers exactly (d2048/k64 0.1823, d2048/k128 0.1878, d2048/k256 0.1853).
- **Pool identity**: `scale_pool_meta.json` 100k = the champion's declared v227 pool
  (`reference_n=100000`, `seed=42`, `band=v227` in `ankh_base_hardneg.pt` meta). Leakage
  asserts pass: pool intersect queries = empty, pool intersect reference = empty
  (`scale_train.py:96-97`). Pinned in `step3/pool_accs.json`. Full pool = the 615,960
  config accessions minus the 7,401 queries minus the 15,000 reference, restricted to the
  534,060 that carry v227 annotations (source `annotation_set c905dffa`).
- **Scoring harness (`knn_confirm.py` verbatim)**: query = 7,401 LAFA targets, reference =
  15,000 t0 subset, cosine top-30 GO-transfer vote over raw reference leaves, cafaeval
  f_micro_w x 9 NK/LK/PK x MFO/BPO/CCO cells (prop=fill, norm=cafa, no_orphans,
  th_step=0.01, PK excludes known). mean9 = mean of the 9 cells. Champion re-scored in the
  SAME harness = the apples baseline (champion head over production L48 base 08234f06).
- **MLflow**: tracked to `http://localhost:5000`, experiment `l10std-regen-step3`
  (8 runs: champion + 7 heads), params + per-cell + mean9 logged.

## Artefacts

- `step3/l10std_step3_report.json` : full machine-readable report (all arms, per-cell,
  leaderboard, recipe, caveats).
- `step3/head.pt` + `step3/head.scaler.npz` : the winner = best L10-std (d4096/k256/100k).
- `step3/head_primary_apples.pt` + `.scaler.npz` : the primary apples arm (d2048/k128/100k,
  matches the champion's dict/top_k).
- `step3/heads/*.pt` + `*.scaler.npz` : every arm's head + its same-stem scaler.
- `step3/pool_accs.json` : pinned 100k pool accessions.
- `step3/l10_prod.npz`, `step3/champ_codes.npz` : cached production L10 embeddings + the
  champion apples codes.
- `step3/pull_embeddings.py`, `step3/l10std_train_eval.py`, `step3/run.log` : the run.

## Honest caveats (do NOT overclaim)

- This is **KNN-retrieval mean9 only**, a CANDIDATE screen number, **NOT** the sealed 0.4063
  reranked headline (a different, downstream measurement). The champion baseline here (0.2150)
  is its retrieval-level score in this harness, not its pipeline number.
- The full-pipeline head-to-head (export -> reranker -> predict -> cafaeval, Steps 4-9 of
  `L10STD_PLAN.md`) is the **conductor's decision**, not run here. On this KNN screen no
  L10-std arm beats the champion, so the honest prior is that a full run will not either;
  but the retrieval level does not always predict the reranked level, so this is a screen,
  not a proof.
- Single seed (42) per arm, matching the champion head meta. The best-vs-primary gap
  (+0.0009) is within plausible seed noise and should not be read as a real ordering.

## Recommendation for the conductor

L10-std is not a champion replacement on this evidence: the MFO collapse at layer 10 costs
more than z-scoring and hyperparameters recover. Two rigorous, publishable options:
1. Keep the L48 champion and document that the debugging-era last-layer choice was in fact
   near-optimal through production (the local-harness +0.0167 explained as an extraction
   confound); OR
2. If a full-pipeline confirmation is still wanted, run Step 4 on the primary apples arm
   (`head_primary_apples.pt`, d2048/k128, same dict/top_k as the champion) for the cleanest
   like-for-like, expecting it to trail. The CCO win of L10-std hints a per-aspect or
   L10+L48 combination could be a separate lever, but single-layer L10-std is not it.
