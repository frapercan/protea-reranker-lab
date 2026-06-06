# ADR D40: IA-aligned reranker training (palanca 1, sample weighting)

**Status:** Accepted as infrastructure; palanca-1 weighting **not adopted**
(probe negative on lk-bpo seed 42, gate STOP). The `ia_weighting` flag ships
opt-in with default `none`; the recommended next step is palanca 2.
**Date:** 2026-06-04

## Context

The reranker is a LightGBM model that reorders (protein, GO) candidates
recovered by KNN over PLM embeddings. The v26/v27-binary champion trains
with `objective: binary` and uniform sample weights. It is reported with two
metrics that diverge by roughly a factor of ten: the lab-internal
candidate-restricted Fmax (no propagation, hard discrimination under heavy
imbalance) and the cafaeval Fmax (`prop=fill`, `norm=cafa`), which is
inflated by ancestor propagation regaling the easy shallow terms (see
`docs/source/metrics.rst`).

The honest, publishable headline is the IA-weighted metric (wFmax / S_min):
Information Accretion (Clark and Radivojac 2013, `IA(t) = -log2 P(t |
parents(t))`) downweights shallow terms that propagation regales and
upweights the deep, informative region where the model's skill actually
shows. The brief (`IA-ALIGNED-RERANKER-BRIEF.md`, sections 3 and 4) defines
three levers ("palancas") to push IA *into* training, not just into
evaluation. This ADR records palanca 1.

## Decision

1. **Palanca 1: IA sample weighting (objective stays `binary`).** Add a
   `ia_weighting` config flag with three modes and wire a per-row `weight=`
   into both `lgb.Dataset` calls in `reranker.py::fit()`:

   - `none` (default): uniform weights. The historical v26/v27-binary path is
     byte-for-byte unchanged (weight array is `None`).
   - `positives`: positive rows get `1 + ia_scale * IA(go)`, negatives stay
     `1.0`. Missing a deep true term costs more than missing a shallow one,
     without disturbing the negative mass set by `neg_pos_ratio=10`.
   - `all`: every row (positive and negative) gets `1 + ia_scale * IA(go)`. A
     false positive on a specific (high-IA) term is also more expensive.

   IA values come from the tracked `datasets/ia/IA-swissprot-exp-v227.txt`
   (v227 SwissProt, verified in F-LAFA-IA.1 / PR #58). Terms absent from the
   table contribute `IA = 0` (neutral weight `1.0`); malformed or negative IA
   values are clamped to `0.0` so a row can never get a negative weight.

2. **Row alignment via staging.** Sample weights must align row-for-row with
   `labels.npy`, which is produced by the streaming staging pipeline (rows
   bucketed by protein, then stable-sorted within a bucket). Staging gained a
   `carry_go_terms` flag that threads `go_term_id` through pass 1 and the sort
   pass and emits a row-aligned `go_terms.npy` per split. The runner loads it,
   maps each row to `IA(go)`, and passes the weight vector to `fit`. The eval
   split never carries go terms (IA weighting applies to training only), so
   train/val and eval use distinct staging schemas. When `ia_weighting=none`,
   staging does not carry go terms and the path is unchanged.

3. **Objective unchanged.** Palanca 1 does not touch the objective. lambdarank
   with IA-encoded gains (palanca 2) and an IA-propagated Fmax `feval` for
   early stopping (palanca 3) are deferred to later slices.

## Expectation management (brief section 5, write this in the thesis)

Aligning training with IA is expected to lift the **wFmax / S_min** (the
honest metric, where there is headroom). The **unweighted cafaeval Fmax**
(the propagation-dominated headline) may move little or dip a touch: the model
trades precision on shallow terms for precision on deep ones, and that number
is near a ceiling set by the frequent-term prior that a tree cannot push. The
publishable claim is the **delta over baseline on the weighted metric**, not
the absolute unweighted Fmax.

Two honesty caveats carry over from F-LAFA-IA.0: the ground truth is
candidate-restricted (optimistic versus a full-pipeline CAFA submission;
robust as a *delta* because both arms share the candidate set), and the lk-bpo
cell is the one where BPO collapses hardest, chosen because IA weighting has
the most headroom there.

## Probe result and gate (lk-bpo, seed 42, bench-v1-K5-v226-lineage)

Probe driver: `experiments/lafa_ia_palanca1/probe_lk_bpo.py`. Three reranker
arms (all v26-binary recipe, all else equal) plus the KNN baseline, each
evaluated with cafaeval under the LAFA protocol (`prop=fill`, `norm=cafa`,
`no_orphans`, `ia=IA-swissprot-exp-v227.txt`). Full table and provenance in
`runs/lafa_ia_palanca1/SUMMARY.md` and
`runs/lafa_ia_palanca1/results.json` (gitignored artefacts).

| arm | cafaeval Fmax | wFmax | S_min | d wFmax vs KNN |
|---|---|---|---|---|
| unweighted | 0.6485 | **0.5945** | 7.057 | +0.1924 |
| ia_positives | 0.6417 | 0.5872 | 7.103 | +0.1851 |
| ia_all | 0.6466 | 0.5890 | 7.089 | +0.1868 |
| knn | 0.4639 | 0.4021 | 11.309 | n/a |

**Verdict: STOP.** The reranker (any arm) beats the KNN baseline massively on
the weighted metric (+0.19 wFmax, far above the F-LAFA-IA.0 reference of
+0.0785), so reranking is clearly worthwhile. But IA sample weighting did
**not** lift wFmax over the plain unweighted reranker: both IA arms dipped
slightly (wFmax 0.5872 / 0.5890 versus 0.5945) and worsened S_min (7.10 / 7.09
versus 7.06; lower is better). The honest read is that on this cell, IA
weighting as a flat per-row `weight=` does not help and marginally hurts.

This is the outcome the brief anticipated as a real possibility (honesty over
headline). Per the gate, we do not fan palanca 1 out to the 24-cell grid.

## Why it did not help (hypothesis) and next step

Binary log-loss with `neg_pos_ratio=10` already concentrates the gradient on
the positive class; multiplying positive weights by `1 + IA` mostly rescales
an already-balanced loss and shifts the decision threshold rather than
teaching the tree to rank deep terms above shallow ones. The cafaeval wFmax is
a per-protein *ranking* metric after ancestor propagation, and a global
sample-weight reweighting is a blunt instrument for a ranking objective.

The recommended next slice is therefore **palanca 2 (lambdarank with
IA-encoded gains)**, not a palanca-1 fanout: discretise IA into integer
relevance buckets for positives and define an increasing `label_gain`, so the
ranker is told *directly* to lift high-IA terms above generic and negative
ones. lambdarank with `group=protein` is already plumbed (`reranker.py`). The
palanca-1 machinery (the `weight=` hook, the IA loader, the row-aligned
`go_terms.npy`) stays in place and remains available as an opt-in flag and as
a building block for palanca 2's relevance encoding.

## Consequences

- `ia_weighting=none` remains the default, so no merged training recipe
  changes silently. IA weighting is opt-in per spec/CLI (`--ia-weighting`).
- Staging carries one extra string column on the IA path only; the default
  path keeps its memory and schema profile.
- Palanca 2 (lambdarank + IA gains) is gated on this probe lifting the
  weighted metric over both the KNN baseline and the unweighted reranker.
