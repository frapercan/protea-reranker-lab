# ProtST Step 7 spec: ProtST as a reranker signal (Path A, final phase)

Author: Francisco Miguel Perez Canales

Read-only spec. No code changed, no job dispatched, no dependency touched while
producing it. This is the file-by-file implementation plan for turning the
validated ProtST text-to-GO signal into ONE MORE evidence signal in the
per-category LightGBM reranker (ADR-D43), targeting the two unwon cells LK-BPO
and PK-BPO. It supersedes the reranker-signal section of
`PROTST_INTEGRATION_PLAN.md` and `PATH_A_PROTST_EXECUTION.md` with verified
current line numbers and a corrected schema version.

## 0. State the implementer must start from (verified)

Two branch corrections that the older plans got wrong. Base every change on
these, not on the local working-tree checkouts.

- **protea-contracts canonical = `origin/main`, `__version__ = "1.4.0"`,
  `SCHEMA_VERSION = "v5"`** (`git show origin/main:src/protea_contracts/feature_schema.py:23`,
  `:__init__.py:54`). Main already carries the `lineage`, `interpro`,
  `classifier`, `self_prior`, `association` families and `feature_docs.py`. The
  local checkout `repositories/protea-contracts` is on `develop` (v4, no
  lineage) and is STALE. The bump in this spec is **v5 -> v6**, contracts
  **1.4.0 -> 1.5.0**, NOT the "v4 -> v5" the prior plans assumed.
- **PROTEA D45 machinery (PR #710/#720) lives only in `origin/develop`**, not in
  the local working tree (local develop is behind at #666). Branch from
  `origin/develop`; the degeneracy check and `feature_family_provenance` cited
  below are on that ref.
- The ProtST backend is live (`protea-backends` main `9fad195`); the canonical
  API `EmbeddingConfig` is **`bd3cd470-e384-4f6a-90cf-574704419373`**
  (model_backend=protst, 512-d `protein_feature`, `normalize=false`,
  `max_length=1022`), holding ~527k reference embeddings + the eval queries in
  `sequence_embedding` (per `PATH_A_PROTST_EXECUTION.md` step 5-6).

## 1. The signal, exactly (the quantity to stamp)

For each (query protein, candidate GO term) the ProtST feature is the quantity
`storage/text_scorer/knn_confirm_text.py:59-71` (`knn_pred`) computes: take the
query's 512-d ProtST `protein_feature`, retrieve the top-30 cosine-nearest
ProtST reference proteins (`K = 30`, `knn_confirm_text.py:27`), cosine-weight a
vote per reference GO term (`votes[go] += cosine` over each neighbour's t0
annotations, dropping non-positive cosines, `:64-68`), and normalise by the
per-query max vote (`v / mx`, `:70-71`). Both banks are L2-normalised before the
dot product (`l2`, `knn_confirm_text.py:44-45`).

Offline this signal beat the champion at kNN (`storage/text_scorer/WRITEUP.md:35-47`):
mean 0.2455 vs champion 0.2150; isolated text contribution nk-BPO +0.062
(leakage-free), lk-BPO +0.072 (the wall), pk-BPO +0.037; survives hard homology.

## 2. Schema diff (protea-contracts, v5 -> v6)

### 2.1 The three columns and why exactly three

Append to `NUMERIC_FEATURES` (`feature_schema.py`, after `association_present`,
main line ~121) three columns, all `float` (LightGBM reads them as numeric, same
convention as `interpro_present`/`classifier_present` which are float 0/1):

| column | dtype | default when producer OFF | default when producer ON, no evidence | meaning |
|---|---|---|---|---|
| `protst_text_score` | float | NaN | NaN | normalised cosine-weighted ProtST kNN vote for the candidate term (Section 1). The primary signal, the thing that beat the champion. |
| `protst_vote_fraction` | float | NaN | 0.0 | fraction of the query's 30 ProtST neighbours that carry the candidate term. A support/coverage feature; decouples "one very close neighbour" from "broad consensus" (mirrors `neighbor_vote_fraction`). |
| `protst_present` | float (0/1) | NaN | 0.0 | whether ProtST contributed to this (protein, go_id) at all. Distinguishes a measured zero from an absent source; lets the booster gate on ProtST coverage (mirrors `interpro_present`/`knn_present`, `feature_schema.py:106-110`). |

Three, not one: the score alone cannot tell the booster whether a low value
means "ProtST looked and found weak support" versus "ProtST had no coverage for
this reference pool", which is exactly the D45 lesson. `protst_present` +
`protst_vote_fraction` carry that, matching the `interpro` family's
score/hit/present triple.

### 2.2 The family

Add to `FEATURE_FAMILIES` (`feature_schema.py`, after the `association` entry,
main line ~234):

```python
"protst_text": ["protst_text_score", "protst_vote_fraction", "protst_present"],
```

Additive, so per the family-aware fingerprint `compute_feature_schema_sha`
(`feature_schema.py:273-298`) every existing booster that selects only its own
families keeps an unchanged sha; only the full `ALL_FEATURES` digest
(`compute_schema_sha`, `:253-270`) and `SCHEMA_VERSION` advance. This is the
same additivity the `classifier`/`self_prior`/`association` families relied on
(comment `feature_schema.py:228-231`).

### 2.3 Version + docs + tests (the contracts-major-bump-cascade landmine)

- `SCHEMA_VERSION = "v5"` -> `"v6"` (`feature_schema.py:23`).
- `__version__ = "1.4.0"` -> `"1.5.0"` (`__init__.py:54`). SemVer note: the
  header docstring says "bumping any field forces a MAJOR release", but the
  actual precedent for additive families is a MINOR bump (1.2.0 -> 1.4.0 added
  interpro + the LAFA six). Follow the precedent: minor bump, additive, no
  removals. Do NOT remove or rename any column (that would break the superset
  rule and every trained booster).
- `__all__` in `__init__.py:55-103` needs NO change: the new columns ride inside
  the already-exported `NUMERIC_FEATURES`/`ALL_FEATURES`/`FEATURE_FAMILIES`
  symbols; no new top-level symbol is introduced. The superset invariant holds
  by construction.
- **feature_docs.py governance (hard gate).** `feature_docs.py` is the single
  source of truth and `scripts/check_feature_docs.py` + `tests/test_feature_docs.py`
  FAIL if a declared column lacks a `FeatureDoc`, or a doc names an undeclared
  column, or a doc's family disagrees with `FEATURE_FAMILIES`
  (`feature_docs.py` module docstring, "Governance"). Add three `FeatureDoc`
  entries to the `_DOCS` list (model them on the `association_total` entry,
  main `feature_docs.py` ~line 790, a `DECLARED_ABSENT` block). Each needs
  `name`, `family="protst_text"`, `summary`, `definition` (prose verified
  against the producer), `producer` (the dotted path of the stamp function
  from Section 3, with the "gated by compute_protst, default False; export
  default emits NaN" marker), `status`, `unit`, `value_range`, `notes`.
  - `status`: **`FeatureStatus.DECLARED_ABSENT`** if the canonical export leaves
    `compute_protst=False` by default (like the LAFA six, honest per ADR-D45);
    or `FeatureStatus.PRODUCED` if the canonical export turns it on. Pick one and
    make the export flag match, so the doc does not lie.
- **Golden digest test.** `tests/test_feature_schema.py` pins
  `compute_schema_sha(ALL_FEATURES)`. Adding three columns changes it; update
  the pinned digest (compute it once, paste it). This is the intended tripwire
  that forces the version bump.
- **Payload flag (contracts side).** Add `compute_protst: bool = False` to
  `PredictGOTermsPayload` (`payloads.py:83-85`, next to `compute_classifier`/
  `compute_self_prior`/`compute_association`) and to `PredictGOTermsBatchPayload`
  (`payloads.py:166-168`). These payloads are in the contracts package, so this
  is part of the same release.
- No `file://` pin anywhere (the cascade landmine). Release contracts 1.5.0 to
  the index, then bump the git/version pin in PROTEA and the lab.

## 3. Producer wiring (stamp-only, both paths, D45-safe)

The producer reads a PRECOMPUTED signal and stamps it onto EXISTING candidate
rows. It never recomputes embeddings and never adds candidate rows. That is what
"stamp-only" means here and it is what keeps the training and eval pools matched
(the classifier pool-mismatch lesson, `project_meta_reranker_direction_2026_06_17.md:44-46`).

### 3.1 The new producer module

Create `protea/core/operations/predict_go_terms/_protst_text.py` (new file,
sibling of `_post_knn_pipeline.py`). It exposes one annotate-only function:

```
apply_protst_text(session, predictions, protst_config_id, t0_annotation_set_id, ...) -> None
```

It: (1) loads the query's ProtST embedding and the ProtST reference bank from
`sequence_embedding` under `protst_config_id` (= `bd3cd470-...`); (2) L2-normalises
both (the config is `normalize=false`, so normalise at read, matching
`knn_confirm_text.py:44`); (3) runs the top-30 cosine kNN vote of Section 1 over
the reference bank using **numpy or FAISS, never pgvector** (hard constraint,
527k > 500k vectors); (4) stamps `protst_text_score` / `protst_vote_fraction` /
`protst_present` onto the matching `(protein_accession, go_term_id)` prediction
rows. Model the stamping on `apply_self_prior`
(`_post_knn_pipeline.py:124-181`, single-column annotate keyed by
`(protein, go_term_id)` with snapshot-invariant go_id-string resolution via
`_resolve_*_go_ids` / `_load_go_id_and_aspect` at `:475-496`). Reference GO
terms MUST be the t0=v227 pre-cutoff set only (leakage, Section 8.3).

### 3.2 Predict path

- Gate it in `_apply_lafa_score_features` (`_post_knn_pipeline.py:80-109`): add
  a `if getattr(p, "compute_protst", False): apply_protst_text(...)` branch
  around line 102, next to the self_prior/association branches.

### 3.3 Export path (same function, verbatim)

The export imports the predict producers directly (`_export_features.py:48-52`),
so writing the ONE function in 3.1 covers both paths. In
`protea/core/training_dump/_export_features.py` (origin/develop):

- `ExportParityFlags` (`:62-77`): add `protst_text: bool = False` and OR it into
  the `.any` property.
- Add `_PROTST_PRODUCER = "protea.core.operations.predict_go_terms._protst_text.apply_protst_text"`
  next to the other producer-name constants (`:67-71`).
- In `apply_export_parity_features` (`:139`): when `flags.protst_text`, call
  `_zero_baseline_family(records, FEATURE_FAMILIES["protst_text"])` (`:196-210`)
  before stamping, exactly as the wired families do (`:180-191`).
- In `build_lafa_family_provenance` (`:213-235`): add the tuple
  `("protst_text", flags.protst_text, _PROTST_PRODUCER)`.

### 3.4 Leaf-record defaults (NaN, not zero: the D45 fix itself)

`protea/core/_leaf_record_builder.py`. Emit NaN for the three columns when no
producer ran (LightGBM native-missing), following the corrected convention that
`_lafa_default_fields` now uses on origin/develop (NaN, `:341-367`), NOT the
old zero-fill:

- Add the three NaN defaults inside `_lafa_default_fields` (or a sibling
  `_protst_default_fields`), modelled on `_lineage_default_fields` (`:320-339`).
- Wire the block into all three call sites, or the T1.8 canonical-column gate
  fails: `make_leaf_record` (`:173`), `_apply_non_knn_defaults` (`:243-254`),
  `build_classifier_only_record` (`:573-612`).

### 3.5 Feature-registry binding (fails loud if unbound)

`protea/core/features/_bindings.py`:

- Add `_PROTST_TEXT_FEATURES = ("protst_text_score", "protst_vote_fraction", "protst_present")`
  near `_ASSOCIATION_FEATURES` (`:277-288`).
- Add a `_protst_text_producer` factory near `_association_producer` (`:162-191`).
- Add a loop over `_PROTST_TEXT_FEATURES` in `_build_feature_to_producer`
  (`:295-328`). The `missing` guard at `:322-327` fails loudly if any declared
  column has no producer binding, so this step is mandatory, not optional.

### 3.6 The compute_protst flag (PROTEA side)

- `protea/core/training_dump/_payload.py:76-78` (`TrainRerankerAutoPayload`): add
  `compute_protst: bool = False` next to `compute_classifier`/`compute_self_prior`/
  `compute_association`.
- `protea/core/_training_dump_loaders.py:383-401` (`_dump_family_provenance`):
  add `protst_text=bool(getattr(p, "compute_protst", False))` when building
  `ExportParityFlags`, so the manifest records produced-vs-absent for the family.

### 3.7 Why this cannot silently zero-fill (the guard chain, #710/#720)

Once `protst_text` is in `feature_family_provenance` as `produced`, the export
degeneracy check catches a null family automatically, in
`protea/core/parquet_export.py` (origin/develop):

- `FamilyProvenance` states `PRODUCED`/`DECLARED_ABSENT` (`:117-142`);
  `feature_family_provenance` on the context (`:180`);
  `_produced_family_columns` maps a PRODUCED family to its columns (`:73-95`).
- `_track_produced_family_values` (`:405-419`, NaN collapses to a sentinel
  `_DEGENERATE_NAN` at `:414`), called from `write_shard` (`:396`).
- `_assert_no_degenerate_families` (`:495+`) FAILS the job if a family recorded
  PRODUCED is constant across the split; invoked on the success path of
  `_stream_train_shards` (`:461`) and `_stream_eval_shards` (`:486`).
- Manifest write records the provenance (`:308-315`, `ManifestV1` uses
  `extra="ignore"`, no contract bump).

Net: if `compute_protst=True` but the bank is missing or every value is
constant, the export dies loudly instead of shipping a fake column; if
`compute_protst=False`, the family is `DECLARED_ABSENT` (all NaN), which the
booster reads as missing. Neither is the D45 silent zero.

## 4. The ProtstTextScorer (served path, ADR-D43)

`protea/core/reranking/scorers.py`. A thin read, no recompute (the producer owns
the computation):

- Add after `AssociationScorer` (`:246`, before `default_scorer_registry`):

```python
class ProtstTextScorer(_KeyedScorer):
    name = "protst_text"
    applies_to = _ALL_CATEGORIES        # text evidence exists for NK too
    def _value(self, candidate: Candidate) -> float | None:
        return _as_float(candidate.get("protst_text_score"))
```

modelled on `LabelEmbeddingScorer` (`:130-142`); `_as_float` (`:54-69`) turns
missing/NaN into `None` so the scorer emits a sparse dict.

- Register it in `default_scorer_registry` (`:248-266`, append after `:265`) and
  add `"ProtstTextScorer"` to `__all__` (`:269-280`). Registration order is the
  canonical MR-2 score-vector column order.

Inputs from `bd3cd470`, in one sentence: given the query's 512-d ProtST
`protein_feature`, retrieve the top-30 cosine-nearest ProtST reference proteins,
cosine-weight-vote their t0 GO terms, normalise by the per-query max vote, and
the producer stamps that normalised vote as `protst_text_score` (with
`protst_vote_fraction` = fraction of the 30 neighbours carrying the term and
`protst_present` = 1 if any voted), which this scorer then reads.

## 5. Combiner retrain + 9-cell eval

There are two combiner realities in the lab. Use both, in order.

### 5.1 Fast head-to-head on the exact sealed frame (schema-driven lambdarank)

The sealed 0.4063-frame number was produced by
`repositories/protea-reranker-lab/results/clean_227230/train_rerank_227230.py`,
a per-category LightGBM **lambdarank** over ~64 raw parquet columns, where
`feature_cols()` (`:57-59`) = all parquet schema names minus an `EXCLUDE` set
(`:30-36`). Therefore a `protst_text_score` column in the exported parquet is
auto-picked-up as a feature with zero code change.

The clean A/B: regenerate the 227->230 train/eval parquet with the three ProtST
columns (Section 6), then run `train_rerank_227230.py` twice:

- arm A (champion baseline): add the three columns to `EXCLUDE` (`:30-36`) =
  the exact sealed champion.
- arm B (champion+ProtST): leave them in.

The ONLY delta between arms is the `EXCLUDE` set, so it is a same-pool same-frame
A/B. Score both with `results/clean_227230/score_rerank_227230.py` through the
same cafaeval harness and compare `f_micro_w` per cell. This gives the gate
YES/NO cheaply, off-platform, on the frozen frame. `comparison.json:2` pins the
sealed frame ("227->230 ... schema_sha 775611822dd9").

### 5.2 Servable combiner + platform eval (promotable path)

For the served pipeline and the "integrated in platform" thesis claim:

- MR-2 shallow combiner: add `"protst_text": ["protst_text_score",
  "protst_vote_fraction", "protst_present"]` to `SCORE_VECTOR_BY_SCORER`
  (`src/protea_reranker_lab/combiner.py:74-82`); do NOT add it to
  `_KNOWN_ONLY_SCORERS` (`:86`, it applies to all categories); update the order
  assertion `tests/test_combiner_mode.py:102-110`. Objective is `binary`
  (calibrated probs for the f_micro_w threshold sweep, `train.py:41`). To
  benchmark before editing, pass `--combiner-columns
  protst_text_score,protst_vote_fraction,protst_present` (`train.py:83`), which
  the dataset reader tolerates for any present column (`data.py:74-77`).
- `infer_active_feature_families` (`native_boosters.py:71-97`) MUST gain a
  `protst_text` branch, or the predict-time `feature_schema_sha` guard
  (`SchemaShaMismatchError`) rejects the model. The sha is
  `compute_feature_schema_sha(infer_active_feature_families())`
  (`native_boosters.py:149-151`).
- Register the trained boosters via `POST /reranker-models/import-by-reference`
  (`README.md:220-227`) with `model_path`, `manifest_sha`, `schema_sha`, `cell`,
  `metrics`.

### 5.3 The 9-cell eval dispatch (POST /jobs)

Operation `run_cafa_evaluation`, queue `protea.evaluations`
(`operation_catalog.py`, `run_cafa_evaluation.py:251`). `RunCafaEvaluationPayload`
fields (`run_cafa_evaluation.py:56-110`): `evaluation_set_id`,
`prediction_set_id`, `rerankers` (nested `{cat: {aspect: reranker_model_id}}`,
`:64-71`), `ia_file` (`:72-82`), `restrict_gt_to_predicted` (`:83-92`),
`softprop` (`:93-100`), `th_step=0.01` (`:101-110`). Dispatch, per
`project_job_dispatch_mechanism_2026_07_11.md`, with a HS256 bearer minted from
`PROTEA_JWT_SECRET` in `worktrees/protea-deploy/.env.local`, body
`{operation, queue_name, payload}`.

Example payload skeleton (fill the reranker ids after registration):

```json
{
  "operation": "run_cafa_evaluation",
  "queue_name": "protea.evaluations",
  "payload": {
    "evaluation_set_id": "<acc27f47 eval set, verify id>",
    "prediction_set_id": "<the ProtST-enriched predict set from Section 6>",
    "ia_file": "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv",
    "th_step": 0.01,
    "restrict_gt_to_predicted": false,
    "rerankers": {
      "nk": {"bpo": "<id>", "cco": "<id>", "mfo": "<id>"},
      "lk": {"bpo": "<id>", "cco": "<id>", "mfo": "<id>"},
      "pk": {"bpo": "<id>", "cco": "<id>", "mfo": "<id>"}
    }
  }
}
```

Run it for both combiner arms (champion, champion+ProtST). Frame: TEST
v227->v230, metric `f_micro_w` only, query_set `acc27f47` (lafa-official-7401).

### 5.4 Stratification (standing norm)

Stratify every reported delta on three axes (length x category(NK/LK/PK x
aspect) x neighbour-identity), CIs on the deltas, per
`feedback_always_stratify_length_category_2026_06_25`. Reuse the offline
neighbour-identity buckets (MMseqs2 hard = twilight+no-hit+remote vs easy) and
the script pattern `storage/text_scorer/stratify_identity.py`. Report the NK-BPO
leakage-free delta beside every LK/PK-BPO delta.

## 6. The materialisation + regeneration prerequisites

Before any eval, the ProtST columns must exist in the frame's parquet and
prediction set. Serial spine:

1. Adopt/complete the ProtST bank under `bd3cd470` (~527k refs + acc27f47
   queries), the in-flight offline run (`PATH_A_PROTST_EXECUTION.md` step 6).
   Verify bank coverage of the acc27f47 candidate reference pool (Section 8.4).
2. `export_research_dataset` (queue `protea.training`), champion embedding
   d8979601, sealed 227->230 frame, `compute_protst=true` -> ProtST-enriched
   training parquet.
3. `predict_go_terms` (queue `protea.predictions`) on acc27f47,
   `compute_protst=true` -> ProtST-enriched prediction set (the eval pool
   carries the columns; matched pool per Section 3).

## 7. The 9/9 gate and honest prior

Gate (pass = all four true), on the sealed frame, `f_micro_w`:

1. LK-BPO champion+ProtST reaches #1 (currently unwon).
2. PK-BPO champion+ProtST reaches #1 (currently unwon).
3. The other 7 cells stay #1 (no regression; do not lose a won cell to a
   ProtST-induced pool shift).
4. The LK-BPO and PK-BPO gains survive length x category x neighbour-identity
   stratification with CIs on the deltas excluding zero, and the NK-BPO
   leakage-free anchor moves the same direction (evidence it is the text signal,
   not leakage).

Honest prior, state it up front. At kNN, ProtST SUBSUMES the champion's BP
(combined approximately equals protst-alone, `WRITEUP.md:103-104`), so a naive
combine buys little over ProtST-alone. The genuinely open question is whether
ProtST ADDS THROUGH THE RERANKER, where many more signals (alignment, taxonomy,
classifier, priors) sit alongside it and the BP wall is precisely where those
are weakest. It might add there (subsume-cosine-kNN-BP does not imply
subsume-reranker-BP) or it might lift PK-BPO without reaching #1 (the modest
pk-BPO +0.037 offline, and PK is the calibration-hardest cell). **An 8/9 outcome
(LK-BPO clears, PK-BPO lifts but not #1) is a valid, reportable result**: the
honest claim becomes "ProtST wins 8/9, dents PK-BPO", not a forced 9/9.

## 8. Risks and landmines

1. **Contracts-major-bump cascade.** Additive only: no rename/removal (superset
   rule), no `file://` pin, update the golden `compute_schema_sha` test, add the
   three `FeatureDoc` entries or `check_feature_docs.py`/`test_feature_docs.py`
   fail. Minor bump 1.4.0 -> 1.5.0, `SCHEMA_VERSION` v5 -> v6. Then bump the pin
   in PROTEA and the lab, regenerate `poetry.lock` properly (do not hand-edit;
   the 2026-07-11 outage was a lock-contention hang).
2. **The D45 silent zero-fill seam.** Emit NaN, never 0.0, when no producer ran
   (Section 3.4); register the family in `feature_family_provenance` so the
   degeneracy check guards it (Section 3.7); keep `feature_docs` status honest
   (`DECLARED_ABSENT` unless the canonical export turns it on).
3. **Leakage in the kNN transfer.** Reference GO terms must be the t0=v227
   pre-cutoff set only. Exclude the query from its own neighbour set. NK is the
   leakage-free anchor (those proteins had no function text at t0); ProtST
   "knows" reference-protein function because it trained on Swiss-Prot text,
   which is a legitimate prior available to all, not answer-leakage, but state
   it to the committee (`WRITEUP.md:99-102`).
4. **The interpro-style config-gap.** The interpro family read zero because its
   source env var was unset at export, indistinguishable from empty. The analog
   here: the ProtST reference bank (`bd3cd470`) and `compute_protst=true` must
   both be present at export AND predict. Unlike interpro, the #710 degeneracy
   check + `DECLARED_ABSENT` provenance now catch a silent gap loudly, but still
   verify bank coverage of the candidate reference pool before dispatch; a
   reference with no ProtST embedding yields `protst_present=0` for its
   candidates (safe, but shrinks coverage).
5. **Pool mismatch.** Stamp-only, annotate existing candidates on BOTH paths; do
   NOT add ProtST candidates (that propose-novel-terms variant is a separate
   follow-up requiring symmetric candidate addition on both paths).
6. **Predict-time schema guard.** `infer_active_feature_families`
   (`native_boosters.py:71-97`) must gain the `protst_text` branch, or the live
   reranker rejects the retrained model with `SchemaShaMismatchError`.
7. **halfvec magnitude.** ProtST `protein_feature` is raw fp32 and `bd3cd470`
   was created `normalize=false`; the producer L2-normalises at read (cosine
   kNN), so the stored magnitude does not affect the score, but confirm the
   halfvec store did not overflow on ingest (#731 guard) before trusting the
   bank.

## 9. Ordered execution and parallelism

Serial spine (each gates the next):
S1 contracts 1.5.0 release (schema + docs + payload flag + golden test).
S2 PROTEA off origin/develop: pin bump + producer module + leaf defaults +
   bindings + export flags + provenance + payload flag + ProtstTextScorer.
S3 deploy PROTEA in a quiet window (standing redeploy auth), watch poetry-lock
   git-dep resolution.
S4 export (compute_protst) -> S5 predict (compute_protst) -> both need S3.
S6 fast A/B (Section 5.1) -> gate YES/NO.
S7 if gate clears: servable combiner + register + platform eval (Section 5.2-5.3).
S8 stratify + write the number on the three surfaces.

Parallelizable:
- P1 ProtST bank materialisation under `bd3cd470` (Section 6.1) is independent of
  S1/S2 code work; run alongside.
- P2 the contracts change (S1) and the PROTEA scorer/producer code (S2) can be
  written on parallel branches; S1 must merge/release before the S2 pin bump.
- P3 the fast offline A/B (5.1) only needs the enriched parquet (S4); it can
  proceed before the servable ProtstTextScorer/combiner registration (S7).
- P4 the stratification script (5.4) can be prepped from the offline pattern in
  parallel with S4-S6.

## 10. Blocking unknowns not resolvable from code

1. **Live contracts pin.** `origin/main` is v1.4.0/v5, but a subagent's venv read
   reported v1.4.0 with `SCHEMA_VERSION="v3"` (inconsistent, likely a stale
   install). Verify what `worktrees/protea-deploy` actually pins before choosing
   the bump base; the expected bump is v5 -> v6.
2. **Bank coverage.** Whether `bd3cd470`'s ~527k references fully cover the
   acc27f47 candidate reference pool was the sizing step of the earlier
   `PROTST_SCALE_SPEC.md`; confirm it is complete, or size the residual gap.
3. **acc27f47 eval-set id.** The evaluation_set_id for acc27f47 (memory records
   `34a634a8` on 2026-06-17) must be reconfirmed against the current DB before
   dispatch.
4. **Frame-parity of `restrict_gt_to_predicted`.** The exact mapping between the
   platform `run_cafa_evaluation` flags (`restrict_gt_to_predicted`, `softprop`)
   and the offline `no_orphans`/`exclude-known` settings
   (`knn_confirm_text.py:77`) needs a one-shot parity check so the platform
   number reproduces the sealed frame.
