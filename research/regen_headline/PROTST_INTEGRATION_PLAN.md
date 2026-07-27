# ProtST full-pipeline integration plan: the 9/9 push at the BP wall

Author: Francisco Miguel Perez Canales

Scope of this document: a read-only execution plan. No code changed, no job
dispatched, no GPU run, no dependency touched while producing it. It carries the
numbered path from today's state to a 9-cell head-to-head (champion vs
champion+ProtST) with an explicit 9/9 gate on the two unwon cells, LK-BPO and
PK-BPO.

## 1. Where we actually stand (verified, cite file:line)

### 1.1 The validated signal (offline, kNN-only)

The ProtST text-to-GO signal is validated at the kNN level and the receipts are
on disk:

- `storage/text_scorer/knn_confirm_text.py:59` (`knn_pred`) and `:74` (`nine`):
  the board-faithful protocol, query 7401 -> reference 15000, cosine top-30,
  cosine-weighted GO vote, `cafa_eval` `f_micro_w`, 9 cells, stratified by
  category. NK is the leakage-free anchor (`knn_confirm_text.py:9-11`).
- `storage/text_scorer/knn_confirm_text_result.json` + `WRITEUP.md:35-50`:
  protst-text mean 0.2455 beats champion d8979601 0.2150; the isolated text
  contribution (protst-text minus esm1b-raw, same ESM-1b base) is positive on
  all 9 cells and largest on BP: nk-BPO +0.062 (leakage-free), lk-BPO +0.072
  (the wall), pk-BPO +0.037; survives hard homology (nk-BPO +0.053, lk-BPO
  +0.044). The BP lift is ProtST-specific: ProTrek reproduces CCO, not BP
  (`WRITEUP.md:52-75`).
- The extractor recipe: `storage/text_scorer/extract_protst.py:8-20`. Tokenizer
  `facebook/esm1b_t33_650M_UR50S`; model `mila-intel/ProtST-esm1b`
  (`trust_remote_code=True`); forward `model.protein_model(input_ids,
  attention_mask)`; read `out.protein_feature` (512-d, fp32); `MAXLEN=1022`.

The three honest caveats stand (`WRITEUP.md:96-106`,
`project_text_evidence_scorer_2026_07_08.md:22,30`): (1) raw kNN, 15k reference,
no reranker, so absolutes are low and the DELTA is the point; (2) ProtST is
supervised on function text, so beating a self-supervised base on
function-transfer is partly by design (NK is clean on the query side, but ProtST
"knows" the reference proteins' function because it trained on their text);
(3) combined approximately equals protst-alone at kNN, i.e. ProtST SUBSUMES the
champion's BP rather than purely complementing it (replace-vs-augment). Caveat
(3) is the crux this plan must resolve through the reranker.

### 1.2 The runtime does NOT have ProtST (and the backend is NOT on develop)

Correction to the task premise: the ProtST backend is NOT merged to
protea-backends develop.

- protea-backends develop tip is `fc411ce` (#36); the last merged PR is #37
  (`2adaaf6`). The ProtST backend lives only on the unmerged branch
  `feat/protst-backend` (commits `0fde550` "ProtST text-aligned embedding
  backend" and `4626048` "pin ProtST checkpoint to a fixed revision"). `git
  merge-base --is-ancestor 0fde550 origin/develop` returns NOT-in-develop.
- On that branch the backend is complete and faithful to the offline recipe:
  `src/protea_backends/protst/__init__.py` implements `EmbeddingBackend`
  (`protein_feature` 512-d, fp32 forward, single whole-protein vector, honours
  only `config.normalize` for L2, tokenizer pinned, checkpoint revision-pinned).
  The entry point `protst = "protea_backends.protst:plugin"` and the extra
  `protst = ["torch", "transformers"]` exist only on the branch (`git show
  feat/protst-backend:pyproject.toml`), not on develop
  (`pyproject.toml:58-62` on develop registers esm/t5/ankh/esm3c only).
- The deploy venv confirms the runtime gap: `protea_backends-0.1.0.dist-info/
  entry_points.txt` registers esm/t5/ankh/esm3c; there is no `protst` package
  directory under the installed `protea_backends`.

Consequence: making ProtST a first-class runtime EmbeddingConfig requires
merging the branch, releasing protea-backends, and re-pinning PROTEA. That is
the dependency-change risk this plan isolates (Section 3).

### 1.3 The reranker seam (how a signal becomes a reranker feature)

- The EvidenceScorer pattern (ADR-D43): `protea/core/reranking/ports.py:57`
  (`EvidenceScorer` Protocol: `name`, `applies_to`, `score(query_ctx,
  candidates) -> {go_id: float}`); adapters in
  `protea/core/reranking/scorers.py` are thin reads of an ALREADY-COMPUTED
  per-candidate signal off the candidate dict / typed `GOPrediction` column
  (e.g. `LabelEmbeddingScorer._value` reads `anc2vec_neighbor_maxcos`,
  `scorers.py:143`). A scorer does NOT recompute; the producer owns the
  computation.
- The signal-store / feature schema: `protea-contracts .../feature_schema.py`.
  `NUMERIC_FEATURES` (`:28`) is the typed column set; `FEATURE_FAMILIES`
  (`:148`) groups columns for on/off training; `SCHEMA_VERSION` (`:23`) is `v4`.
  A new feature column bumps `SCHEMA_VERSION` and forces a protea-contracts
  major release plus booster retrain (`feature_schema.py:1-14`, `:253`).
- The combiner is the shallow per-category meta-learner over the score VECTOR
  (`ports.py:87`, `combiners.py`); the trained per-category combiner is what
  reaches the board number.

### 1.4 The pool-mismatch lesson (do not repeat it)

`project_meta_reranker_direction_2026_06_17.md:44-46` (CANDIDATE-POOL MISMATCH):
when the classifier ADDED full-vocab candidates on the predict path but the
training export only carried KNN candidates, the combiner scored the added rows
out-of-distribution and f_micro_w DROPPED. The rule this bakes in: the ProtST
feature must be produced on the SAME candidate pool the combiner trains AND
evals on. The export path and the predict path must both stamp it.

### 1.5 Reproducibility anchors (the sealed frame)

- Frame: TEST v227 -> v230 (6 months); metric `f_micro_w` only; the single
  frame behind the sealed 0.4063.
- Query set: `acc27f47` (lafa-official-7401), the 7401 eval queries used in the
  offline confirm (`knn_confirm_text.py:29`, meta.json accs).
- Eval inputs: OBO `go-basic.obo` (Sep_2025), `IA.tsv`, `groundtruth_*` +
  `groundtruth_terms_of_interest.txt` (`knn_confirm_text.py:20-26`); `norm=cafa`,
  `prop=fill`, `th_step=0.01`, `no_orphans=True`, PK excludes known
  (`knn_confirm_text.py:77`).
- Leakage asserts: NK-BPO is the clean anchor (NK proteins had no function text
  at t0=v227); every reported BP delta must show the NK-BPO delta beside the
  LK/PK deltas so the committee sees the leakage-free number.
- Determinism: seed-averaged producers where seeds exist; L2-normalise both
  query and reference banks before cosine (matches `knn_confirm_text.py:44`
  `l2`); pin the ProtST checkpoint revision (already pinned on the branch,
  `4626048`).

## 2. The integration mechanism (exact)

ProtST enters the reranker as a NEW per-candidate text-to-GO feature, produced
identically on the export and predict paths, read by a thin EvidenceScorer,
consumed by retrained per-category combiners.

### 2.1 The feature

For each (query protein, candidate GO term), the ProtST feature is the
cosine-weighted vote strength that ProtST-kNN gives that term, exactly the
quantity `knn_confirm_text.py:64-71` computes (top-30 ProtST neighbours of the
query, cosine-weighted sum over each neighbour's reference GO terms, normalised
by the max vote). Concretely, mirror the `anc2vec_neighbor` family:

- `protst_text_score` (float): the normalised ProtST vote for the candidate
  term. NaN when ProtST proposed no evidence for it (LightGBM native-missing).
- `protst_vote_fraction` (float): fraction of the query's ProtST neighbours that
  carry the candidate term.
- `protst_present` (0/1): whether ProtST contributed to this (protein, go_id),
  so a true zero is distinguishable from an absent source (same convention as
  `interpro_present`, `feature_schema.py:106-110`).

New family in `feature_schema.py:FEATURE_FAMILIES` named `protst_text`; three
columns appended to `NUMERIC_FEATURES`; `SCHEMA_VERSION` bumps `v4 -> v5`
(protea-contracts major release; additive, so existing boosters keep their
family-aware sha per `compute_feature_schema_sha`, only the full digest advances,
`feature_schema.py:229-234`). [platform-dispatch: contracts release + PROTEA pin]

### 2.2 The producer

A new producer computes the three columns from: the query's ProtST embedding,
the ProtST reference bank (per-reference 512-d), and the reference GO
annotations. It runs in the feature-enrichment stage of BOTH:

- `export_research_dataset` (training parquet), and
- `predict_go_terms` (live prediction rows),

so training and eval pools match (Section 1.4). Stamp-only first: score the
candidates the ankh-KNN pool already proposed (no ProtST-added candidates). This
keeps the pool matched, isolates the reranker-ADD question cleanly, and avoids
the classifier-style pool mismatch. The propose-novel-terms upside (ProtST adds
candidates the ankh-KNN misses, its NK/BP value) is a SEPARATE follow-up that
requires adding candidates on BOTH paths together. [offline producer logic ported
from `knn_confirm_text.py`; platform-dispatch to run at scale]

### 2.3 The scorer + combiner

- New `ProtstTextScorer(_KeyedScorer)` in `scorers.py`, `applies_to =
  _ALL_CATEGORIES` (text evidence exists for NK too), `_value` reads
  `protst_text_score` (thin read, no recompute). Register it in
  `default_scorer_registry` (`scorers.py:250`). [platform-dispatch]
- Retrain the per-category combiners (protea-reranker-lab) over the score vector
  extended by the ProtST column, focus LK-BPO + PK-BPO. Train on the ProtST-
  enriched export, eval via the `run_cafa_evaluation` `rerankers` payload on the
  ProtST-enriched predict set. [platform-dispatch: lab train + PROTEA eval]

## 3. Materialising ProtST at scale: Path A vs Path B (recommendation)

The producer needs ProtST vectors for the full reference pool (every reference a
candidate can come from) and the 7401 query set. Today only a 15k-reference /
7401-query subset exists (`ref_protst.npy`, `query_protst.npy`). Two ways to get
the full banks.

### Path A: wire the runtime backend

Merge `feat/protst-backend` -> protea-backends develop, release protea-backends,
bump the PROTEA `poetry.lock` pin, redeploy; register a ProtST `EmbeddingConfig`;
`compute_embeddings` runs it at scale into `SequenceEmbedding`.

- Pro: platform norm (`feedback_integrate_in_platform_not_adhoc`), provenance-
  versioned config, KNN + feature production fully on-platform, permanent.
- Con: the poetry-lock git-dependency risk. The 2026-07-11 outage
  (`project_deploy_migration_incident_2026_07_11.md:14-16`) was a `poetry
  install` hang from git-dep lock contention (a stale `poetry update` held the
  lock), site down ~20 min. A protea-backends bump re-enters that exact hazard.
  Also GPU hours for the full ~527k-protein ProtST extraction, plus a redeploy
  window.

### Path B: offline extraction into a real EmbeddingConfig

Extend `extract_protst.py` to the full reference pool + the 7401 queries; load
the banks into a real ProtST `EmbeddingConfig` + `SequenceEmbedding` rows via the
`store_embeddings` path (not loose npy).

- Pro: no dependency change, no `poetry.lock` risk, no redeploy; reuses the
  validated extractor byte-for-byte; still lands in the DB signal-store with
  provenance (store-first charter), just produced by the pinned offline recipe
  rather than the registered plugin.
- Con: bypasses the "produced by the registered backend" norm; a drift risk
  between the offline extractor and the eventual plugin (mitigated: the plugin
  is a literal port of this extractor, `protst/__init__.py` docstring says so).

### Recommendation: Path B first, Path A to make it permanent

Run Path B to get the reranker number without touching dependencies. Do NOT pay
the poetry-lock / redeploy risk until the signal has proven it ADDS through the
reranker and clears the #1 bar (Section 5). Once the gate clears, execute Path A
in a quiet window (no concurrent heavy DB job, per the incident checklist
`project_deploy_migration_incident_2026_07_11.md:34-42`) to make ProtST a
first-class runtime backend for the served pipeline and the thesis's
"integrated in the platform" claim.

Halfvec / embedding_scale note (#731): the fp16 `halfvec` store overflows on
large magnitudes; `_compute_embeddings_helpers.py:36` `fetch_embedding_scale` and
`compute_embeddings.py:533` guard it. ProtST `protein_feature` is raw fp32 and
its magnitude is unverified, BUT the offline confirm L2-normalises before cosine
(`knn_confirm_text.py:44`) and the backend honours `config.normalize`. Set
`normalize=True` on the ProtST `EmbeddingConfig`: unit vectors are always
halfvec-safe and no `embedding_scale` is needed. Verify the raw magnitude once
(Section 6) before committing, so the decision is measured, not assumed.

## 4. The numbered path (each step marked)

1. [offline, read-only] Magnitude + coverage check: measure `protein_feature`
   raw magnitude on the existing `query_protst.npy` / `ref_protst.npy` (numpy,
   on disk) to settle normalize-vs-embedding_scale; enumerate the reference-pool
   coverage gap (15k present vs the full v227 pool the candidate set draws from).
   Output: the exact extraction job spec (accession list, batch size, expected
   GPU hours). This is the FIRST action (Section 6).
2. [offline GPU, later] Extend `extract_protst.py` to the full v227 reference
   pool + the 7401 acc27f47 queries; L2-normalise; write banks + meta. (GPU run,
   NOT part of this scoping doc.)
3. [platform-dispatch] Load the banks into a ProtST `EmbeddingConfig` +
   `SequenceEmbedding` rows via `store_embeddings` (Path B), `normalize=True`.
4. [contracts] Add the `protst_text` family (3 columns) to `feature_schema.py`,
   bump `SCHEMA_VERSION v4 -> v5`, release protea-contracts, bump the PROTEA +
   lab pins (additive; family-aware shas unchanged).
5. [PROTEA] Add the ProtST producer to BOTH the export and predict
   feature-enrichment paths (stamp-only, matched pool); add `ProtstTextScorer`
   to `scorers.py` and register it.
6. [platform-dispatch] Re-export the training dataset in the sealed frame with
   the ProtST columns populated (`export_research_dataset`, `protea.training`).
7. [platform-dispatch] Re-predict the 7401 acc27f47 frame with the ProtST
   producer on (predict path), so the eval pool carries ProtST columns.
8. [lab] Retrain the 9 per-category combiners over the score vector + ProtST
   column; keep a champion (no-ProtST) combiner set trained on the SAME re-export
   as the control, so the head-to-head is same-pool same-frame.
9. [platform-dispatch] `run_cafa_evaluation` (`rerankers` payload) on the ProtST-
   enriched predict set for both combiner sets -> the 9-cell head-to-head
   (champion vs champion+ProtST), `f_micro_w`.
10. [analysis] Stratify each cell by length x category x neighbour-identity
    (MMseqs2 hard = twilight+no-hit+remote vs easy), CIs on the per-cell deltas,
    NK-BPO delta reported beside LK/PK as the leakage-free anchor. Apply the 9/9
    gate (Section 5).
11. [conditional] If the gate clears, execute Path A (backend merge + release +
    redeploy in a quiet window) to make ProtST a runtime backend; then surface
    the result on the three surfaces (thesis / interface / Sphinx). If it does
    not clear, record the honest negative (Section 5) and, as the teed-up
    upside, run the propose-novel-candidates variant (Section 2.2) which requires
    adding ProtST candidates on both paths together.

## 5. The 9/9 gate and the honest prior

Gate (pass = all four true):

1. LK-BPO champion+ProtST > LK-BPO of every other config in the sealed frame,
   i.e. reaches #1 on `f_micro_w` (LK-BPO is one of the two currently unwon
   cells).
2. PK-BPO champion+ProtST reaches #1 `f_micro_w` (the second unwon cell).
3. The other 7 cells stay #1 (no regression; especially do not lose an
   already-won cell to a ProtST-induced pool shift).
4. The LK-BPO and PK-BPO gains survive stratification (hold on the hard-homology
   tail, not a homology proxy) with CIs on the deltas excluding zero, and the
   NK-BPO leakage-free anchor moves in the same direction (evidence the lift is
   the text signal, not leakage).

The honest prior (state it up front, do not let the number surprise the
committee): at kNN, ProtST SUBSUMES the champion's BP (combined approximately
equals protst-alone, `WRITEUP.md:104`, caveat 3). The open, genuinely-uncertain
question is whether ProtST ADDS THROUGH THE RERANKER. Two reasons it might, and
one it might not:

- Might add: the reranker carries many more signals than raw cosine kNN
  (alignment, taxonomy, classifier, priors), so "subsumes cosine-kNN-BP" does
  not imply "subsumes the reranker's BP"; the combiner can use ProtST as one
  more calibrated column that lifts LK/PK-BPO where the other signals are weak
  (the BP wall is precisely where they are weak).
- Might add: ProtST is orthogonal to homology on the hard tail
  (`WRITEUP.md:46-47`), which is where the reranker currently has least to work
  with on BP.
- Might NOT clear the bar: subsume-at-kNN plus the modest pk-BPO delta (+0.037)
  means PK-BPO in particular may lift without reaching #1 (PK is the calibration-
  hardest cell historically); and #1 is an absolute-rank bar, not a
  positive-delta bar.

So the deliverable is a clean YES/NO on each of LK-BPO and PK-BPO reaching #1,
with the subsume caveat stated, not a headline that assumes the kNN lift
transfers. A partial result (LK-BPO clears, PK-BPO lifts-but-not-#1) is a valid
and reportable outcome; it would make the honest claim "ProtST wins 8/9, dents
PK-BPO" rather than 9/9.

## 6. The first concrete action

Read-only, no GPU, no dependency, on artefacts already on disk:

Measure the ProtST `protein_feature` magnitude and the reference-pool coverage
gap. On `storage/text_scorer/query_protst.npy` and `ref_protst.npy` (numpy):
report min/max/mean L2 norm and per-dim max abs (to decide normalize=True vs an
`embedding_scale`, Section 3 note), and diff the 15k reference accessions
(`ref_meta.json`) against the full v227 reference pool the acc27f47 candidate set
draws from, to size the extraction job (Step 2). Output: a one-page
`storage/text_scorer/PROTST_SCALE_SPEC.md` with the halfvec decision and the
exact extraction job spec.

This unblocks the materialisation path (Step 2-3) with a measured, not assumed,
halfvec decision and a concrete extraction size, and it touches nothing live.
</content>
</invoke>
