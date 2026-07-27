# BP failure vs protein length / truncation (cheap re-analysis, frozen data, CPU)

**Question.** Is the BP failure (and DeepGO-SE's separability advantage over the deployed
reranker) concentrated in LONG / TRUNCATED proteins, where the champion encoder d8979601
(learned k-WTA over Ankh-base MEAN-pool, `max_length=2048`) is both diluted and C-terminus
truncated? If yes, re-computing the non-truncating chunk-attention codes (f4df03fe) for long
queries is justified; if length is flat, it is not.

**Method.** No new training, no DB, no job dispatch. Lengths from
`storage/text_scorer/ident/query.fasta` (7,401 targets; 216 = 2.9% are >2048 TRUNCATED,
13.7% >1024). Reused the pool / anchor / propagation machinery of
`storage/deepgose_kwta/deepgose_measure.py` (obo `prop=fill`, IA weights, reachable-tail AUC)
over the frozen SE ensemble (`se_scores_{LK,PK}.npz`, avg) and the deployed reranker
predictions. Delivered-f is the per-protein IA-weighted-F of the deployed system at the cell's
micro-optimal threshold (a per-protein stratification proxy, not the board f_micro_w).
Buckets: <=512, 512-1024, 1024-2048, >2048 (TRUNCATED).
Script: `storage/length_strat/length_stratify.py`; JSON: `storage/length_strat/length_stratification.json`.

## LK-BPO (523 proteins scored by SE; tau*=0.40, micro-f 0.36126)

| bucket | n | n% | true-IA% | mean terms | delivered f | prec | rec | unreach-FN frac | n_sep | SE auc | RR auc | SE-RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <=512 | 348 | 66.5 | 66.8 | 17.2 | 0.3516 | 0.371 | 0.411 | 0.240 | 343 | 0.873 | 0.835 | +0.038 |
| 512-1024 | 140 | 26.8 | 26.2 | 17.7 | 0.3585 | 0.347 | 0.443 | 0.159 | 138 | 0.855 | 0.818 | +0.037 |
| 1024-2048 | 31 | 5.9 | 6.1 | 18.6 | 0.2289 | 0.275 | 0.267 | 0.265 | 30 | 0.846 | 0.809 | +0.036 |
| >2048 | 4 | 0.8 | 0.9 | 21.8 | 0.1106 | 0.163 | 0.139 | 0.491 | 4 | 0.870 | 0.870 | -0.001 |

## PK-BPO (4,402 proteins; tau*=0.03, micro-f 0.23463)

| bucket | n | n% | true-IA% | mean terms | delivered f | prec | rec | unreach-FN frac | n_sep | SE auc | RR auc | SE-RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <=512 | 2447 | 55.6 | 55.7 | 18.3 | 0.2042 | 0.199 | 0.269 | 0.268 | 1645 | 0.621 | 0.483 | +0.138 |
| 512-1024 | 1395 | 31.7 | 31.2 | 17.9 | 0.1842 | 0.178 | 0.248 | 0.270 | 914 | 0.621 | 0.497 | +0.124 |
| 1024-2048 | 443 | 10.1 | 10.2 | 18.3 | 0.1702 | 0.165 | 0.240 | 0.313 | 257 | 0.677 | 0.508 | +0.170 |
| >2048 | 117 | 2.7 | 2.8 | 18.8 | 0.1808 | 0.187 | 0.252 | 0.288 | 69 | 0.637 | 0.547 | +0.090 |

## Isolated >2048 TRUNCATED slice (controlling for term count)

| | LK (n=4) | PK (n=117) |
|---|---|---|
| trunc mean true terms | 21.8 | 18.8 |
| trunc delivered f | 0.1106 | 0.1808 |
| <=2048 delivered f | 0.3461 | 0.1942 |
| term-count-matched control f | 0.3588 | 0.2024 |
| trunc unreach-FN frac | 0.491 | 0.288 |
| <=2048 unreach frac | 0.219 | 0.273 |
| term-count-matched unreach frac | 0.243 | 0.282 |
| trunc SE auc / RR auc | 0.870 / 0.870 | 0.637 / 0.547 |
| <=2048 SE auc / RR auc | 0.867 / 0.829 | 0.626 / 0.490 |

## Findings (honest magnitudes)

1. **Long proteins carry NO disproportionate IA weight.** True-IA% tracks n% almost exactly in
   both cells (e.g. PK >2048 = 2.7% of proteins, 2.8% of IA), and mean true-term count is FLAT
   across buckets (~17-19). The premise that long proteins have outsized IA leverage is NOT
   confirmed here.

2. **SE's separability advantage over the reranker does NOT grow with length.** LK: flat
   (+0.037/+0.037/+0.036, then n=4 noise). PK: +0.138 / +0.124 / +0.170 / **+0.090** -- the
   truncated >2048 slice has the SMALLEST SE-over-reranker advantage of any bucket, and the
   reranker's own tail AUC is actually HIGHEST there (0.547). The chunk-attn hypothesis (a
   better-separating rep helps most where the mean truncates) is not supported; if anything the
   truncated slice is where the deployed rep separates BEST.

3. **Delivered-f / headroom is length-flat in the robust cell.** PK delivered-f is essentially
   flat (0.204 / 0.184 / 0.170 / 0.181 -- the truncated slice slightly RECOVERS vs 1024-2048),
   and unreachable-FN fraction is flat (0.27-0.31). LK shows a mild dip in the 1024-2048
   dilution zone (f 0.23 vs 0.35 baseline), but that is a ranking/separability effect in the
   NON-truncated zone, and the >2048 LK bucket is only n=4 (unusable).

4. **The truncated >2048 slice is not systematically worse.** PK (n=117, robust): trunc f 0.181
   vs term-count-matched control 0.202 -- a ~0.02 gap, with IDENTICAL unreachable fraction
   (0.288 vs 0.282). So the tiny gap is not a reachability/incompleteness effect from
   truncation; it is not larger than term-count noise. LK truncated is n=4 and cannot support a
   truncation claim (SE only scored 523 LK proteins).

## Verdict

**LENGTH-FLAT. The BP failure is NOT concentrated in long/truncated proteins -> the chunk-attn
recompute is NOT worth it.** Truncation touches 216 proteins (2.9%, 2.8% of BP IA mass); within
the robust PK slice they are not systematically worse than term-count-matched short proteins,
their unreachable fraction is identical, and DeepGO-SE's separability edge (the signal a
non-truncating rep would exploit) is WEAKEST exactly there. The real BP headroom is
unreachability (0.24-0.31 of true IA never in the pool) and separability, both roughly
length-flat -- consistent with the prior atlas (length only weakly negative for LK, flat for PK)
and with the SEPARABILITY-not-reachability wall. Recomputing/swapping f4df03fe chunk-attention
codes for long queries is not justified by the data.
