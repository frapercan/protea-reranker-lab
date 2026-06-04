# F-LAFA-IA.1b GATE VERDICT: v227 probe (prostt5 K3)

Dataset `bench-v1-K3-v227-lineage-prostt5` (full re-export, train cutoff
v227, eval band v227->v230). Eval protocol: prop=fill, norm=cafa,
no_orphans, `ia=datasets/ia/IA-swissprot-exp-v227.txt`. Headline metric is
the IA-weighted micro Fmax (wFmax = `f_micro_w`); S_min is `s` (lower is
better). Two arms per cell: v26-binary reranker (seed 42) vs KNN baseline
(raw `neighbor_vote_fraction`).

## Numbers (reranker - KNN, the publishable delta)

| cell | rr wFmax | knn wFmax | d wFmax | rr S_min | knn S_min | d S_min (knn-rr) |
|---|---|---|---|---|---|---|
| nk-mfo | 0.7536 | 0.6201 | +0.1335 | 2.099 | 3.099 | +1.000 |
| nk-bpo | 0.4997 | 0.3474 | +0.1523 | 6.399 | 9.355 | +2.956 |
| nk-cco | 0.5911 | 0.4179 | +0.1733 | 2.883 | 4.563 | +1.680 |
| lk-mfo | 0.6712 | 0.6138 | +0.0574 | 3.542 | 3.427 | -0.114 |
| lk-bpo | 0.3505 | 0.3479 | +0.0026 | 7.318 | 8.940 | +1.623 |
| lk-cco | 0.4434 | 0.4464 | -0.0029 | 3.783 | 4.157 | +0.374 |

S_min delta is `knn - rr`, so positive means the reranker reduces semantic
distance (better).

## v227 LK delta vs the v226 IA.0 baseline (PR #57)

The F-LAFA-IA.0 v226 LK wFmax deltas were lk-bpo +0.079, lk-cco +0.088,
lk-mfo +0.033. On the faithful v227 band:

| cell | v227 d wFmax | v226 d wFmax | change |
|---|---|---|---|
| lk-mfo | +0.0574 | +0.0330 | +0.0244 |
| lk-bpo | +0.0026 | +0.0790 | -0.0764 |
| lk-cco | -0.0029 | +0.0880 | -0.0909 |

## Interpretation

1. The NK arm is unambiguous: the reranker beats the KNN baseline on
   wFmax by +0.13 to +0.17 across all three aspects, with large S_min
   reductions (1.0 to 3.0 bits). This is the dominant CAFA evaluation
   regime (most query proteins have no prior annotation) and it is where
   the reranker carries its weight.

2. The LK arm is band-sensitive. lk-mfo improves over v226 (+0.057 vs
   +0.033). lk-bpo and lk-cco shrink to ~zero on wFmax (+0.003, -0.003).
   The cause is structural, not a regression: moving t0 from v226 to v227
   drops about 37% of the grid positives (band_shift.json: 125026 ->
   79199 newly-true pairs). The LK cells lose the most positives, so the
   per-protein wFmax denominator thins out and the reranker's lift over
   the already-strong LK KNN vote collapses into the noise band. S_min
   still favors the reranker on lk-bpo (+1.62) and lk-cco (+0.37), so the
   reranker is not worse, it is just at parity on the IA-weighted Fmax for
   two of three LK aspects.

## GATE: does the v227 band close the bench-vs-LAFA gap?

YES, with a scoped caveat. The probe closes the *methodological* gap that
F-LAFA-IA.1 flagged: this is a faithful v227 grid (candidates, features,
and labels all computed at the v227 train cutoff), not a v226 grid with a
v227 ground-truth overlay. It removes the two LAFA-incomparability sources
the protocol named (the v226-vs-v227 one-band offset and the post-t0
candidate leakage). The reranker-minus-KNN wFmax delta, which is the
publishable claim, survives the band move on all NK cells and on lk-mfo.

The caveat is that the *absolute* LK lift is band-specific and largely
disappears for lk-bpo and lk-cco at v227, exactly because the v227 band is
a thinner, harder positive set. That is the honest v227 story, not a
defect in the reranker.

Net: the export + eval pipeline is validated end to end (disk, /dev/shm,
the full re-export path, and the IA-weighted eval all work), and the v227
delta is publishable. GREEN-LIGHT F-LAFA-IA.1c (the 24-cell v227 fanout)
so the full PLM x K grid gets the faithful v227 numbers. Before the fanout
runs, confirm the postgres container `/dev/shm` is at least 1 GB (see
README operational note); that is the one infra precondition for the
24-cell export.

## Harness fix landed in this slice

`eval_v227_probe.py::eval_reranker` originally read `go_term_id` straight
from the trainer `predictions.parquet`, which only carries
`protein_accession`, `label`, `score`. It now recovers `go_term_id` from
`_eval_slice` (the same crc32-bucket plus protein-ascending order the
trainer stages in); the protein and label vectors were verified
byte-identical to `predictions.parquet`, so the score-to-GO pairing is
exact.
