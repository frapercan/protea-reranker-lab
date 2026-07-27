# From-scratch headline regeneration (author decision 2026-07-11: "regenerar TODO desde cero")

Goal: one PRESENT platform receipt that supersedes the three drifted numbers
(0.4004 served / 0.4063 offline champion / 0.408 LAFA harness). Re-derive the
whole pipeline on the champion config, then harmonize thesis + UI + Sphinx to
the single regenerated number.

Champion frame: embedding_config `d8979601-ea59-4de1-9c16-21036ed67c36`,
ontology_snapshot `aeb07c36-17db-4b0b-a560-a68376998476`, k=30,
train_versions [160,165,170,175,180,185,190,195,200,205,211,215,220,225,227],
test_versions [230]  (= v227->v230, Sep2025->Mar2026).

Dispatch via `POST /jobs` on the platform (never ad-hoc). NEEDS A QUIET WINDOW:
step 1 uses the same GPU as live /annotate embeddings; steps strain DB I/O on
the 113GB go_prediction table. Fire when the API is idle (overnight).

## Step 1 - export_research_dataset  (queue protea.training, GPU/RAM heavy, hours)
Payload (from job fd166542, the sealed export):
```json
{"k":30,"output_name":"regen-from-scratch-train227-test230",
 "test_versions":[230],
 "train_versions":[160,165,170,175,180,185,190,195,200,205,211,215,220,225,227],
 "compute_taxonomy":true,"compute_alignments":true,"compute_classifier":true,
 "compute_self_prior":true,"compute_association":true,
 "embedding_config_id":"d8979601-ea59-4de1-9c16-21036ed67c36",
 "ontology_snapshot_id":"aeb07c36-17db-4b0b-a560-a68376998476"}
```
Produces train.parquet + eval.parquet + a new Dataset row. Verify: `GET /datasets`
shows the new name, row counts > 0, schema_sha recorded.

## Step 2 - reranker-lab training, per category  (offline, host, protea-reranker-lab)
For each of NK / LK / PK: pull the new dataset, train the LightGBM booster,
import back as a RerankerModel (POST /reranker-models/import or import-by-reference).
The sealed board used 3 per-category rerankers:
lk=83d758fe-11c3-4d75-a93d-3ebf8295b36f  nk=94f0030e-834f-48ad-937f-f1ef3b1a93b1
pk=66c5a25f-a6bd-4b5e-be84-fe9b730540ca  (these get REPLACED by the freshly trained ones).
Verify: 3 new RerankerModel rows, each feature_schema_sha matching the live pipeline.

## Step 3 - predict_go_terms  (queue protea.predictions)
On the champion query set, producing a fresh PredictionSet + GOPredictions.
(Reference predict job 13b1cae1; query_set_id + annotation_set_id from the frame.)
Verify: new prediction_set row, GOPrediction count > 0.

## Step 4 - run_cafa_evaluation  (queue protea.evaluations, isolated)
Payload (from job b41beff9, the board), with the NEW reranker + prediction ids:
```json
{"ia_file":"/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv",
 "toi_file":"/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_terms_of_interest.txt",
 "interpro_graft":true,
 "reranker_id_lk":"<new lk>","reranker_id_nk":"<new nk>","reranker_id_pk":"<new pk>",
 "evaluation_set_id":"6e41eb5b-df02-4400-95c5-9cef6c9029ed",
 "prediction_set_id":"<new pset from step 3>",
 "interpro_ipr2go_file":"/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/interpro2go_test/ipr2go_prop.json",
 "interpro_graft_weight":null,
 "interpro_protein2ipr_file":"/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/interpro2go_test/protein2ipr.json"}
```
Produces the EvaluationResult board (results jsonb, 9 cells NK/LK/PK x MFO/BPO/CCO).
Headline = mean-of-9 f_micro_w. This is the authoritative PRESENT receipt.

## Step 5 - harmonize (only after step 4's receipt exists)
Write the regenerated number + the #1-in-N/9 count to all three surfaces:
thesis (frontmatter/abstract.tex, ch6/ch7/appendix_d, currently 0.408),
UI (apps/web/lib/book.ts HEADLINE + CHAPTER_ZERO + PILLARS, currently 0.4063),
Sphinx (docs/source/overview.rst + results, currently 0.4063). Also dump the
board to storage/clean_227230/comparison.json so the receipt is a present file.
If LAFA-harness genuinely differs from cafaeval, state which is headline + explain
the other on each surface.

## Phase 1a code-switch design (scoped 2026-07-11 from the live tree)

The switch is SMALL: ~50 base features are ALREADY typed-column-based (their jsonb
mirror is dead weight, no reader change needed). Only the 6 LAFA scalars + IA still
live in the blob. Concrete changes (all in worktrees/protea-deploy tree paths;
implement in a fresh worktree worktrees/signal-store-switch, PR base develop):

SINGLE WRITE PATH: core/operations/predict_go_terms/_common.py
- `_row_from_prediction` (:220-251): :235-236 already writes ~50 typed cols via
  `_STORE_FLOAT_KEYS`. :240 `build_feature_jsonb(row)` + :241 `_attach_lafa_features`
  write the redundant blob (6 LAFA + IA JSONB-only). CHANGE: add the 6 LAFA keys +
  IA to the typed-column build (values already on `pred`; per-key `_clean_float(pred.get(...))`),
  then REMOVE the `build_feature_jsonb`/`_attach_lafa_features`/`row["features"]=` write.
- `build_feature_jsonb` lives in infrastructure/orm/models/embedding/go_prediction_features.py.
- store worker: core/operations/predict_go_terms/_store.py:100-103 (no change).

SINGLE READ PATH: core/operations/_run_cafa_helpers.py:124-126
- `_record_from_pred` reads the 6 LAFA + IA from the blob. CHANGE: read them from
  typed columns via `getattr(pred, col)` (base features already read typed at :122-123).

BLOCKER - IA has NO typed column: migration f9b2c1a4d7e0 added only the 6 LAFA cols.
`IA` (7th key, _common.py:127, _run_cafa_helpers.py:93, _blob_provenance.py ia family)
is blob-only. MUST add an additive migration `ia` Float nullable (down_revision
f9b2c1a4d7e0) BEFORE the switch, and write/read it too.

GUARD - core/operations/predict_go_terms/_blob_provenance.py fingerprints the
classifier/self_prior/association/IA blob families (the 0.3462/0.220 incident guard).
Repoint it to the typed columns (or the in-memory candidate dict) as part of the switch;
warn-only, but must not silently go blind.

DOCSTRINGS (stale, fix): core/reranking/scorers.py:24-26 and
infrastructure/orm/models/embedding/go_prediction.py:125-129 still call the blob the
read source.

SHA SAFE: feature_schema_sha is NAME-based (protea-contracts feature_schema.py
compute_feature_schema_sha), blind to blob-vs-column; moving LAFA to typed columns moves
NO sha -> rerankers not invalidated. (Note IA is NOT in NUMERIC_FEATURES; adding a typed
`ia` may need a contracts review if it should join the schema - check before assuming.)

NOT droppable yet: the DROP COLUMN features + reclaim is a FOLLOW-UP migration AFTER the
switch is deployed + verified (new predictions carry typed LAFA/IA, nothing reads the blob).
Old prediction sets have LAFA/IA in blob only (typed cols NULL) - acceptable, they are
superseded by 1b's regeneration; do NOT re-evaluate an old pset for LAFA after the switch.

ORDER: 1a-i add `ia` migration -> 1a-ii code-switch PR (write+read typed, drop blob write,
guard, docstrings) -> CI -> merge -> deploy -> verify a fresh prediction populates typed
LAFA/IA and nothing reads `features` -> 1a-iii follow-up DROP COLUMN + reclaim (window,
frees 74GB) -> 1b regenerate on the lean store.

## Present-state receipt audit (2026-07-11) - the three numbers this supersedes
- 0.4004 SERVED composite: evaluation_result row 2026-07-02, pset a0cc322f,
  rerankers lk 83d758fe / nk 94f0030e / pk 66c5a25f. mean-of-9 = 0.4004 exactly. PRESENT.
- 0.4063 offline champion: storage/clean_227230/comparison.json, ABSENT locally.
- 0.408 thesis: LAFA submission harness, ABSENT locally.
