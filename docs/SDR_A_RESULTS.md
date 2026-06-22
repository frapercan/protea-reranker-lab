# SDR-A readout-1: overlap-vs-GO-semantic correlation (RESULTS)

The cheapest decisive test of the `sparse.pdf` hypothesis (Appendix A.2, readout-1)
on PROTEA's own benchmark, leakage-clean. The hypothesis under test:

> an interpretable set-overlap in a sparse space (k-WTA + Tanimoto) can be at least
> as good a proxy for biological similarity as a distance in a dense space (cosine).

## Setup

- **Representation pool:** the frozen v227 (GOA-227, Sep 2025) ProtT5 reference
  bundle (`~/Thesis2/storage/protea-frozen-v227-2025-09-04`): 574,627 proteins,
  1024-dim mean-pooled `Rostlab/prot_t5_xl_half_uniref50-enc` embeddings, raw
  (un-normalised). This is the SELECT-window t0 reference pool; nothing post-cutoff
  (t1) is ever read, so the readout is leakage-clean by construction.
- **Ontology snapshot (t0):** `go-basic.obo` release 2025-07-22 (matching the v227
  cutoff), 48,165 terms; is_a + part_of closures.
- **Biological-similarity ground truth:** GO semantic similarity over the DAG.
  Information content is derived from the propagated annotation corpus (Resnik IC);
  pair similarity is Resnik (IC of the most-informative common ancestor) and, as a
  cross-check, Lin (`2*IC(MICA)/(IC(a)+IC(b))`).
- **Dense arm (baseline):** cosine similarity on the raw embeddings.
- **Sparse arm:** k-WTA binarisation (top-k coordinates by magnitude) + Tanimoto
  set-overlap `T = |A & B| / (2k - |A & B|)`, sweeping `k in {32, 64, 128}`.
- **Readout:** Spearman correlation of (cosine vs GO) and (Tanimoto_k vs GO) over a
  large sample of protein pairs.
- **Scale:** 5,000 sampled annotated reference proteins, 200,000 distinct protein
  pairs, seed 42.

## Numbers

Spearman rho vs GO semantic similarity (higher = better proxy for biology):

| arm | Resnik | Lin |
|:-|:-|:-|
| dense cosine | 0.3153 | 0.3032 |
| SDR Tanimoto k=32 | 0.2287 | 0.2051 |
| SDR Tanimoto k=64 | 0.2485 | 0.2413 |
| SDR Tanimoto k=128 | 0.2551 | 0.2520 |

All correlations are highly significant (p < 1e-200; the comparison is about
effect size, not significance). The two semantic metrics agree.

## Gate verdict: NEGATIVE (on this readout, this sweep)

Under the strict gate (best Tanimoto within 0.01 of cosine), the hypothesis does
**not** hold at readout-1: dense cosine is the stronger proxy for GO semantics
(0.315 vs the best Tanimoto 0.255 at k=128), a gap of about 0.06 Spearman.

Honest caveats that keep this from being a flat refutation:

- **The gap closes monotonically as k grows** (0.229 -> 0.249 -> 0.255 at
  k=32/64/128). The sparse.pdf sweep tops out at k=128; the trend suggests larger
  active sets narrow the gap further. A wider k sweep is the obvious follow-up
  before the program is judged.
- This readout binarises the **existing** ProtT5 geometry by magnitude. It tests
  whether a sparse *post-transform* of a dense space preserves biological signal,
  not whether a natively sparse representation (SDR-B SimHash, SDR-C learned
  k-sparse autoencoder, SDR-D structure-derived pharmacophore) would. The negative
  here is specifically about magnitude-k-WTA over ProtT5, the cheapest rung.
- Pairs mix all three GO aspects; Resnik MICA across aspects is naturally 0 (the
  roots differ), so the readout is effectively within-aspect for any informative
  pair.

## Decision

Do **not** advance to a full SDR-A k-NN arm on `/benchmark` on the strength of this
readout. Record the result cleanly as a negative-at-readout-1 with the monotonic-in-k
caveat. Cheapest sensible next step before abandoning the sparse axis: extend the k
sweep upward (256, 512) on this same correlation harness; if Tanimoto reaches cosine
parity at a tractable k, proceed to the k-NN arm, otherwise pivot to the natively
sparse rungs (SDR-B/C/D) rather than k-WTA over a dense space.

## Reproduce

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
poetry run python scripts/run_sdr_a_correlation.py
```

The runner defaults the `bundle` and `obo` paths to the frozen v227 pool and the t0
ontology snapshot, so no flags are needed; pass `semantic-metric lin` (with the
usual leading double dash) to swap the Lin metric for the default Resnik one. The
MLflow runbook (`docs/source/runbooks/mlflow.rst`) has the full env setup.

Tracked in MLflow under experiment `sdr-a-correlation` (runs `sdr-a-resnik`,
`sdr-a-lin`): params, the per-k Spearman metrics, the summary table, and the
scatter plot. Core math lives in `src/protea_reranker_lab/sdr.py` (unit-tested in
`tests/test_sdr.py`); the runner is `scripts/run_sdr_a_correlation.py`.
