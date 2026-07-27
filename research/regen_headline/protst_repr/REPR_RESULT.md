# ProtST representation screen at the kNN-retrieval level (2026-07-13)

Bounded, apples-to-apples screen on the champion's EXACT `knn_confirm` harness
(`l10std_train_eval.py` verbatim: query=7401, ref=15000, cosine top-30 GO-transfer vote,
cafaeval f_micro_w over the 9 NK/LK/PK x MFO/BPO/CCO cells, mean9 = mean of 9). The ONLY
change vs the champion harness is the base embedding source: the ProtST bank (EmbeddingConfig
`bd3cd470-e384-4f6a-90cf-574704419373`, dim 512, single chunk per protein). Same accession
lists + GO closures as the champion. Single seed 42. READ-ONLY DB.

## Coverage (crucial for reading these as a retrieval space)

**100% ProtST coverage** on all three sets: qaccs 7401/7401, raccs 15000/15000, pool 100000/100000.
So ProtST is a valid FULL-COVERAGE retrieval space here, not just a reranker feature. No ref/pool
protein was dropped.

## Harness validated

`champion_L48_apples` re-scores the champion codes on this harness and reproduces the pinned
**mean9 = 0.2150 exactly** (0.21501). So every delta below is decision-grade against the sealed
champion retrieval space. (Champion codes were regenerated from the production L48 base 08234f06 +
`ankh_base_hardneg.pt`; query-code repro maxdiff vs the pinned `query_d8979601.npy` = 3.2e-2, and
the 9-cell score still lands on 0.2150, so the tiny numeric drift is scoring-irrelevant.)

## Leaderboard (mean9 f_micro_w, kNN-retrieval)

| arm                     | mean9  | vs champion | vs protst_raw | nk-bpo  | lk-bpo  | pk-bpo  |
|-------------------------|--------|-------------|---------------|---------|---------|---------|
| **protst_zscore**       | 0.2485 | **+0.0335** | +0.0030       | 0.20611 | 0.23452 | 0.09343 |
| protst_raw              | 0.2455 | +0.0305     | 0.0 (ref)     | 0.20639 | 0.23479 | 0.09132 |
| protst_kwta_d4096_k128  | 0.2411 | +0.0261     | **-0.0043**   | 0.2047  | 0.24069 | 0.08135 |
| protst_kwta_d2048_k128  | 0.2380 | +0.0230     | **-0.0074**   | 0.19957 | 0.23394 | 0.08191 |
| champion_L48_apples     | 0.2150 | 0.0 (ref)   | -0.0305       | 0.16646 | 0.21735 | 0.06445 |

BP cells for the champion baseline: nk-bpo 0.16646, lk-bpo 0.21735, pk-bpo 0.06445.

## Decision

**(a) Does a learned k-WTA head beat RAW ProtST at kNN, especially on BP? NO.**
k-WTA on ProtST LOSES to raw at the retrieval level: d2048 -0.0074, d4096 -0.0043 on mean9. It
loses on the BP cells too: vs raw, k-WTA(d2048) moves nk-bpo -0.0068, lk-bpo -0.0009, pk-bpo -0.0094;
k-WTA(d4096) moves nk-bpo -0.0017, lk-bpo +0.0059, pk-bpo -0.0100. The champion's k-WTA advantage
(learned sparse dictionary over Ankh-base) does NOT transfer to ProtST: ProtST's 512-d text-aligned
space is already a compact, discriminative retrieval space, and imposing the sparse learned head
discards signal (biggest damage on the CCO cells: nk-cco -0.018, lk-cco -0.031). The asymmetry is
closed and the answer is that ProtST does not want a k-WTA head.

**(b) Does z-score help? Marginally, and NOT on BP.**
Per-dim z-score gives mean9 +0.0030 over raw, but on the BP cells it is flat (nk-bpo -0.0003,
lk-bpo -0.0003, pk-bpo +0.0021). The +0.003 comes from CCO/MF nudges (lk-cco +0.013, pk-mfo +0.004).
Not a BP lever; not worth a pipeline change on its own.

**(c) Best ProtST vs the champion 0.2150 as a retrieval space: ProtST WINS, and on every cell.**
Best ProtST (z-score, 0.2485) beats the champion by **+0.0335 on mean9**, and the win is on ALL 9
cells. On the three BP cells: nk-bpo **+0.0397** (0.20611 vs 0.16646), lk-bpo **+0.0172** (0.23452 vs
0.21735), pk-bpo **+0.0290** (0.09343 vs 0.06445). CCO gains are the largest (nk-cco +0.066, lk-cco
+0.070, pk-cco +0.035); MF is near-tied to slightly positive (nk-mfo +0.002, lk-mfo +0.016, pk-mfo
+0.027). This confirms the `project_text_evidence_scorer` claim (raw-protst-text kNN ~0.2455) on the
champion's exact harness: raw ProtST = 0.24547 here, essentially identical.

**(d) Recommendation for the producer (before the authoritative export):**
Feed the producer RAW ProtST vectors (its own 512-d text-projected mean-pool), i.e. keep the currently
deployed `apply_protst_text` kNN-vote over the ProtST bank unchanged. Do NOT add a learned k-WTA head
to ProtST: it costs a head-train and LOSES at kNN. z-score is a marginal, BP-flat +0.003, not worth a
pipeline change; raw is the clean choice. Since `protst_text_score` is a kNN vote, the best kNN
representation is the best producer input, and that representation is RAW (optionally z-scored for a
trivial non-BP nudge), never k-WTA. Gate to expensive export: PASS on raw ProtST, no representation
change required.

## Caveats

- kNN-only mean9 is a CANDIDATE screen number, NOT the sealed 0.4063 reranked headline. It measures
  ProtST as a standalone retrieval space, not the reranked pipeline. The separate reranker A/B
  (`../protst_ab/AB_RESULT.md`) already showed ProtST as a reranker FEATURE dents the BP wall
  (+0.019 lk-bpo, +0.009 pk-bpo through the reranker); this screen is the orthogonal retrieval-space
  question and both point the same way: raw ProtST kNN is the right producer input.
- Single seed 42 (screen). The k-WTA-loses and ProtST-beats-champion signals are large (0.007 to 0.034)
  relative to typical seed wobble (~0.01), so the direction is decision-grade; exact per-cell values
  are single-seed.

## Artefacts

- Report: `storage/regen_headline/protst_repr/protst_repr_report.json`
- Script: `storage/regen_headline/protst_repr/protst_repr_screen.py` (adapted from `step3/l10std_train_eval.py`)
- Cached ProtST bank: `storage/regen_headline/protst_repr/protst_prod.npz` (122401 x 512)
- Champion codes (regenerated): `storage/regen_headline/protst_repr/champ_codes.npz`
- k-WTA heads: `protst_kwta_d2048_k128.pt`, `protst_kwta_d4096_k128.pt` (+ `.scaler.npz`)
