# Native reranker IA + `_p` de-risk experiment

Offline lab experiment: does adding the offline champion's two missing feature
groups (IA + the `_p` variants) to the native "lean" feature set improve the
offline validation score? Run over the existing SELECT 220->227 parquet, no serve
code touched, no live DB.

## Headline verdict: PROCEED (ship IA; drop sp_present; no DAG-propagation)

Adding **IA** (information accretion) to the native lean 30-feature set improves
`f_micro_w` in **every** category on the validation frame, biggest on NK (the
category where the native reranker trails the offline champion most).

| category | lean (control) | lean+IA | Δ f_micro_w | Δ eval AUC |
|----------|---------------:|--------:|------------:|-----------:|
| NK       | 0.6082         | 0.6181  | **+0.0099** | +0.0036    |
| LK       | 0.6296         | 0.6363  | **+0.0067** | +0.0013    |
| PK       | 0.3605         | 0.3636  | **+0.0031** | +0.0013    |
| **MEAN** | 0.5328         | 0.5393  | **+0.0065** |            |

IA importance (gain): NK rank 5/32, LK rank 6/32, **PK rank 4/32** (gain 4.0M).
`sp_present` importance: **0.0 in all three categories** (dead feature).

## The key finding (the brief's `_p` hypothesis was a misreading)

The brief asked us to test "IA + the `_p` propagated variants (knn_p / clf_p / sp_p,
scores propagated UP the GO DAG)". Reading the offline champion's reference
implementation (`worktrees/native-boosters-lab/fullgo/ensemble_seal.py`, the driver
that produced `feature_spec.json` and the sealed 0.391) shows that the champion's
`_p` suffix is a **PRESENCE flag** (`1.0 if term in that stream`), NOT a
DAG-propagated score. In `rows_for()` the 16 features map as:

```
knn_p  <- (1.0 if t in knn_stream)        # presence, not propagation
clf_p  <- (1.0 if t in classifier_stream)
sp_p   <- (1.0 if t in self_prior_stream)
assoc_p<- (1.0 if t in assoc_stream)
```

The native lean 30-feature set **already carries** `knn_present`,
`classifier_present`, and `association_present`. The champion's `lfreq` =
`log1p(t0-pool frequency)` is mirrored by the native `go_term_frequency`. So the
only champion features the native lean set genuinely **lacks** are:

- **IA** = `IA[candidate_go_id]` (information accretion weight) -- genuinely missing
- **sp_present** = `self_prior_score > 0` -- the one missing presence flag

The real, correct test is therefore **LEAN + IA + sp_present**, which is what we ran.

## Method

1. Pulled `datasets/fullgo-union-SELECT-160-220-227-v5/{train,eval}.parquet` from
   MinIO (train 74.0M rows, eval 8.03M rows; schema verified with pyarrow). Columns
   include `protein_accession`, `go_term_id` (already a `GO:` string),
   `classifier_score`, `self_prior_score`, `vote_count`,
   `association_total/cross/present`, etc.
2. Trained 3 per-category LightGBM boosters for the control (30 lean features) and
   the treatment (32 = lean + IA + sp_present), one category at a time, identical
   params, seed 42, streaming row-group batches into per-category float32 matrices
   (peak RSS well under box RAM; PK = 67.5M rows held as float32). IA = lookup on
   `IA.tsv` (39,906 terms, 99.8% coverage on this frame). sp_present = `(sp > 0)`.
3. The control reproduced the documented lean baseline AUC exactly
   (NK 0.9070 / LK 0.9210 / PK 0.8758 vs documented 0.9069 / 0.9211 / 0.8766),
   validating the pipeline.
4. Scored both variants with cafaeval (cafaeval-protea fork) using the offline-seal
   no-TOI recipe (`ia=IA, prop=fill, norm=cafa, no_orphans=True, th_step=0.01`),
   collapsing duplicate (protein,term) candidate rows by MAX. f_micro_w per category
   = mean over the 3 namespaces of the per-ns max.

## Per-namespace (lean+IA)

- NK: bpo 0.5357 / cco 0.6083 / mfo 0.7103  (control 0.534 / 0.5923 / 0.6982)
- LK: bpo 0.5851 / cco 0.6196 / mfo 0.7041  (control 0.585 / 0.6178 / 0.686)
- PK: bpo 0.2392 / cco 0.4288 / mfo 0.4228  (control 0.2347 / 0.4258 / 0.421)

The NK lift is broad (CCO +0.016, MFO +0.012); IA helps wherever rare/deep terms
matter, which is the NK regime.

## Caveats

- **Optimistic magnitudes**: boosters early-stop on this exact eval split and the
  no-TOI recipe evaluates a smaller term universe than the official 7401 test frame,
  so the absolute f_micro_w sit far above the 0.477/0.482/0.215 test numbers. The
  **deltas** (control vs treatment, identical code + eval) are the de-risk signal,
  not the absolute magnitudes.
- The DAG-propagation interpretation of `_p` was NOT tested as a separate booster
  because reading the champion code showed it is not what `_p` means; building that
  plumbing would be speculative and (on PK) memory-risky for zero expected gain,
  since the presence flags it would replace are already present and already low-gain.

## Recommendation

**Proceed to serve-side code for IA only.** Add a per-candidate
`IA = IA[go_term_id]` lookup to the native reranker feature builder (IA.tsv ships in
`lafa_t0_Sep_2025/`), retrain the native boosters, and expect part of the NK/LK gap
toward the offline 0.391 champion to close. **Do not** add `sp_present` (dead) and
**do not** build DAG-propagation plumbing for the `_p` features (the champion's `_p`
= presence, already covered by the native `*_present` flags).

## Artefacts (this directory)

- `train_lean_ia.py` -- control + treatment trainer (lean / lean_ia)
- `train_lean_ia_prop.py` -- literal DAG-propagation trainer (kept for the record; NOT
  run, see caveat)
- `apply_and_score.py` -- apply boosters + cafaeval f_micro_w
- `lean/`, `lean_ia/` -- boosters + `summary.json` + `validation_score/result.json`
- `summary.json` -- machine-readable verdict + numbers
