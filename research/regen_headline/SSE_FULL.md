# SSE Full-Corpus: Mode A (generator) + Mode B (reranker feature)

Sparse Semantic-Entailment scaled to the full experimental corpus, all three aspects. Two-tower comparison (Mode A) and reranker-feature integration (Mode B), true-frame f_micro_w under the temporal gate (train<=v225, blind eval v227-v230).

## Config run

- Corpus: cooc_experiment/generator_frames raw_esm2_3b (ESM2-3B mean, 2560-d); eval proteins held out of ALL training: 7002
- SDR N=2048 protein-bits=128 ensemble K=5 epochs<=14 axiom-weight=0.8; MIN positives/term=40
- MFO: corpus 56227 prot, vocab 808 terms, axioms nf1=1047/nf4=79, mean val AUC 0.9785
- BPO: corpus 57375 prot, vocab 3970 terms, axioms nf1=7548/nf4=1741, mean val AUC 0.9365
- CCO: corpus 56104 prot, vocab 653 terms, axioms nf1=799/nf4=387, mean val AUC 0.9747

## Mode A -- SSE as generator vs the deployed two-tower (PK/LK x aspect)

Best top-k delta of adding SSE out-of-pool proposals to the deployed two-tower+reranker pool (true-frame f_micro_w vs reproduced anchor). Positive = SSE adds true terms the two-tower missed.

| cell | anchor | SSE sep-AUC | rr sep-AUC | best deltaB | rand-order | matched-vol | verdict |
|---|---|---|---|---|---|---|---|
| PK-mfo | 0.24831 | 0.6099 | 0.6296 | -0.0748 (top5) | -0.07494 | -0.15546 | WORSE (-0.0748) |
| PK-bpo | 0.14351 | 0.6196 | 0.4815 | -0.01029 (top5) | -0.01036 | -0.07303 | WORSE (-0.01029) |
| PK-cco | 0.2677 | 0.6185 | 0.6882 | -0.02519 (top5) | -0.02533 | -0.16621 | WORSE (-0.02519) |
| LK-mfo | 0.42243 | 0.8599 | 0.7923 | -0.02156 (top5) | -0.03734 | -0.22436 | WORSE (-0.02156) |
| LK-bpo | 0.31323 | 0.8415 | 0.8285 | 0.00309 (top25) | -0.01043 | -0.15546 | BETTER (+0.00309) |
| LK-cco | 0.37042 | 0.9439 | 0.8373 | 0.02279 (top5) | -0.03397 | -0.17519 | BETTER (+0.02279) |

_All three PK cells (honest `-known` frame): SSE generation is WORSE than the two-tower and tracks its own random-order control (no ranking signal; the loss is volume, matched-vol far worse). The small LK-BPO/LK-CCO Mode-A positives (+0.003/+0.023) are in the no-`-known` frame = the same memorization artifact seen in Mode B, not a genuine generation gain._

## Mode B -- SSE as a reranker feature (retrained per-category LightGBM)

| cell | deployed anchor | baseline | +sse | delta(+sse) | bootstrap CI95 | +sse_shuffled | verdict |
|---|---|---|---|---|---|---|---|
| pk-mfo | 0.2416 | 0.24831 | 0.25152 | 0.00321 | [-0.01124, 0.01761] (fpos 0.668) | -0.01075 | NO LEVER (tie, +0.00315 CI[-0.01124, 0.01761]) |
| pk-bpo | 0.14351 | 0.14351 | 0.14227 | -0.00124 | [-0.00539, 0.00296] (fpos 0.2875) | -0.00224 | NO LEVER (tie, -0.00124 CI[-0.00539, 0.00296]) |
| pk-cco | 0.2655 | 0.2677 | 0.27573 | 0.00803 | [-0.00405, 0.01947] (fpos 0.901) | 0.00171 | NO LEVER (tie, +0.00789 CI[-0.00405, 0.01947]) |
| lk-mfo | 0.41998 | 0.42243 | 0.52241 | 0.09998 | [0.0612, 0.13843] (fpos 1.0) | 0.03891 | SUSPECT-MEMORIZATION (+0.09945 CI[0.0612, 0.13843]; LK has NO -known, so SSE re-derives already-known annotations; shuffled control +0.03891 -> NOT a genuine lever) |
| lk-bpo | 0.31323 | 0.31323 | 0.31905 | 0.00582 | [-0.01204, 0.02369] (fpos 0.752) | 0.01137 | NO LEVER (tie, +0.00603 CI[-0.01204, 0.02369]) |
| lk-cco | 0.35724 | 0.37042 | 0.47311 | 0.10269 | [0.07267, 0.1356] (fpos 1.0) | -8e-05 | SUSPECT-MEMORIZATION (+0.10278 CI[0.07267, 0.1356]; LK has NO -known, so SSE re-derives already-known annotations; shuffled control -8e-05 -> NOT a genuine lever) |

### SSE feature importance in the retrained reranker (+sse variant)
- nk: sse_score gain rank 1/71 (gain 4191244)
- lk: sse_score gain rank 1/71 (gain 2896607)
- pk: sse_score gain rank 2/71 (gain 769457)

## Leakage & honest-frame analysis (Mode B)

The +sse VALIDATION proxy jumped hugely (nk-bpo 0.35->0.60, lk/pk mfo/cco +0.07..+0.08), but validation proteins are NOT eval-held-out, so their sse_score is IN-FIT (SSE saw their labels). On the CLEAN held-out TEST (eval v227-v230, eval proteins excluded from SSE training) the gains collapse:

- **PK cells (honest frame, `-known` excluded): NO lever.** pk-mfo +0.003 (CI crosses 0), pk-bpo -0.001 (CI crosses 0), pk-cco +0.008 (CI crosses 0). SSE's +0.135 BP separability edge does NOT convert even when the reranker can weight it (sse_score ranks #1 by gain but adds nothing at the operating point).

- **LK-MFO/LK-CCO show +0.10 but are SUSPECT-MEMORIZATION, not levers.** LK groundtruth has NO `-known` exclusion, so SSE re-derives a protein's ALREADY-KNOWN annotations from its embedding and those count as true positives. The shuffled-feature control confirms it: lk-cco shuffled +0.0000 (all signal is the protein-term SSE value = memorization), lk-mfo shuffled +0.039 (largely distributional). LK-BPO +0.006 is beaten by its own shuffled control (+0.011) = pure noise.

- **BP specifically: SSE-as-a-feature is NOT a lever.** pk-bpo -0.001 (CI[-0.005,+0.003]); lk-bpo tie/noise. The campaign's first BP lever did NOT materialize.

## Reproducibility note

SSE is fully reproducible from frozen inputs (seeds fixed, models + meta saved under storage/sse_full/models/). The deployed two-tower's SVD/projection basis was never persisted, so its candidate generation is not re-derivable; SSE as a candidate source is auditable where the two-tower is not, independent of the head-to-head f_micro_w outcome.

## Verdicts

**(1) Mode A -- SSE better or worse than the two-tower as a candidate source (per aspect PK):**
  - PK-MFO: WORSE (-0.0748)
  - PK-BPO: WORSE (-0.01029)
  - PK-CCO: WORSE (-0.02519)

**(2) Mode B -- is SSE-as-a-feature a lever (esp BP):**
  - pk-mfo: NO LEVER (tie, +0.00315 CI[-0.01124, 0.01761])
  - pk-bpo: NO LEVER (tie, -0.00124 CI[-0.00539, 0.00296])
  - pk-cco: NO LEVER (tie, +0.00789 CI[-0.00405, 0.01947])
  - lk-mfo: SUSPECT-MEMORIZATION (+0.09945 CI[0.0612, 0.13843]; LK has NO -known, so SSE re-derives already-known annotations; shuffled control +0.03891 -> NOT a genuine lever)
  - lk-bpo: NO LEVER (tie, +0.00603 CI[-0.01204, 0.02369])
  - lk-cco: SUSPECT-MEMORIZATION (+0.10278 CI[0.07267, 0.1356]; LK has NO -known, so SSE re-derives already-known annotations; shuffled control -8e-05 -> NOT a genuine lever)
