# Path A: ProtST as a first-class signal in the method

Author choice (2026-07-12): do not leave ProtST as an offline nohup script.
Make it an established signal: a real backend in the runtime, a config created
through the API, embeddings computed (or blessed) through the platform, and the
reranker signal as a proper feature family with a producer.

## What was already true (verified 2026-07-12)

- The ProtST backend IS merged to `protea-backends` **develop** (PR #38,
  merge `757999b`, CI green). It is a faithful, revision-pinned
  (`e8f6eb21`) port of the validated offline extractor
  `storage/text_scorer/extract_protst.py`: same model
  (`mila-intel/ProtST-esm1b`), same tokenizer (`facebook/esm1b`), same
  forward (`protein_model -> protein_feature`, 512-d), fp32 compute. So
  the offline-materialized vectors should be bit-identical to the
  backend's output.
- The earlier "backend may be on an unmerged branch" note was a
  squash-merge diff artefact; the backend content IS in develop.
- `_dispatch_embed` in PROTEA routes `model_backend` through
  `_resolve_backend` -> `protea-backends` plugin `embed_chunks` (the local
  `_embed_*` are only parity-test references). So the runtime uses the
  plugin; no PROTEA embed re-implementation is needed.
- `_resolve_backend` is pure entry-point discovery (no private whitelist).
- The config-creation validator has a HARDCODED whitelist:
  `VALID_BACKENDS = frozenset({"esm","esm3c","t5","ankh","auto"})` in
  `protea/services/_embeddings_validation_helpers.py:20`. `protst` must be
  added here (this is why the API POST returned 422).
- PROTEA pins `protea-backends` at `branch = "main"`
  (`resolved_reference 5b454e81`), and protst is on develop NOT main -> the
  pin must move to a main ref that includes protst.
- `contracts` main = v1.4.0 (reconciled superset, has interpro/lineage);
  `protea-backends` main already pins `contracts@main`. The backend does
  not touch `feature_schema`, so the contracts pin is not a blocker.

## The chain (minimal-risk variant)

1. DONE - backend on protea-backends develop.
2. protea-backends: minimal release to **main** = cherry-pick ONLY the
   protst backend (diff of `757999b`) onto a release branch off
   `origin/main`; keep `contracts@main` pin; PR base main; CI green
   (`coauthor-guard / scan`); merge. Ships only the backend, not the 30
   unrelated develop commits (defer that promote to a deliberate release).
3. PROTEA (base develop): add `"protst"` to `VALID_BACKENDS`; bump the
   `protea-backends` pin to the new main ref (poetry.lock
   `resolved_reference` = the outage-risk git-dep step; regenerate the lock
   properly, do not hand-edit); add an `_embed_protst` parity shim ONLY if
   a parity test requires one; local CI; PR base develop; merge.
4. Deploy PROTEA (conductor, standing redeploy auth): watch the git-dep
   resolution for the 2026-07-11 poetry.lock contention; keep the site up.
5. Create the ProtST EmbeddingConfig via the API (now `model_backend=protst`
   passes): `layer_indices=[0]`, `layer_agg=mean`, `pooling=mean`,
   `normalize=false`, `max_length=1022`, `embedding_scale=1.0`.
6. Materialize: the offline run under SQL-config `594701e0` is completing in
   parallel (~527k). Bit-identity spot-check (backend vs offline on a
   sample). If identical (expected), either (a) run `compute_embeddings` via
   POST /jobs under the API config for a canonical reproducible
   materialization, or (b) bless/re-key the offline rows to the API config.
   Prefer (a) for reproducibility; the ~3h GPU is cheap.
7. Reranker signal (already first-class by design): `protst_text` feature
   family (3 cols, contract schema v4->v5) + stamp-only producer on
   export+predict + a `ProtstTextScorer` EvidenceScorer (ADR-D43) + retrain
   per-category combiners (focus LK-BPO + PK-BPO) -> run_cafa_evaluation 9
   cells. 9/9 gate: LK-BPO AND PK-BPO reach #1 f_micro_w (sealed
   v227->v230), other 7 hold, deltas survive length x category x
   neighbor-identity stratification with CIs>0, NK-BPO anchor moves same way.

## Execution log (2026-07-12)

- Step 2 DONE: protst on protea-backends main `9fad195` (PR #39). Needed a branch-protection
  context fix (`coauthor-guard / scan` -> `scan`; the guard still runs + is required; no --admin).
- Step 3 DONE: PROTEA PR #732 merged to develop (`d7c3e51`): `VALID_BACKENDS += protst` + pin bump to
  `9fad195` (lock diff minimal, no new deps, no re-resolution). No `_embed_protst` shim (no test needs it).
- Step 4 DONE: deployed develop `d7c3e51` (`deploy.sh origin/develop --no-build`, keyring null to avoid
  the poetry hang). `poetry install` used the fixed lock (no re-resolution -> no outage). Verified:
  /health 200, public 307, the offline extraction SURVIVED, and the runtime now discovers the `protst`
  plugin (`entry_points('protea.backends')` -> ProtstBackend). No git-dep contention.
- Step 5 DONE: created the canonical API config **`bd3cd470-e384-4f6a-90cf-574704419373`**
  (model_backend=protst, layer_agg=mean, normalize=false, max_length=1022, embedding_scale=1.0). The API
  accepted protst (422 gone). JWT needs claims sub+role=operator+iat+exp (iat was the missing one).
- Step 6 PLAN (author CHOSE "adopt" over platform-recompute, 2026-07-12: bit-identical + faster +
  reuses the in-flight run; the spot-check validates the production path). Deferred until the offline
  extraction finishes, to avoid GPU contention: let the offline
  run complete under SQL-config `594701e0`; bit-identity spot-check (compute_embeddings a few seqs under
  `bd3cd470` via POST /jobs vs the offline rows); if identical, re-key
  `UPDATE sequence_embedding SET embedding_config_id='bd3cd470...' WHERE embedding_config_id='594701e0...'`
  and drop the SQL config `594701e0`. Result: 527k under the canonical config, backend-reproducible.

## Step 7 progress (2026-07-12)

- Spec written: PROTST_STEP7_SPEC.md (schema v5->v6, 3 cols, ProtstTextScorer kNN top-30, D45-safe).
- Contracts v1.5.0 MERGED to main `21f300f` (PR #43): protst_text family (protst_text_score /
  protst_vote_fraction / protst_present, DECLARED_ABSENT NaN default), NUMERIC 70->73, ALL 75->78,
  golden sha bcb5453c5a0e, compute_protst flag on the predict payloads. Additive, __all__ superset.
- PROTEA producer+scorer PR IN FLIGHT (agent): pin bump to contracts 21f300f + _protst_text producer
  (stamp-only, D45 NaN default, feature_family_provenance) + ProtstTextScorer (thin read) + export/predict
  wiring. Base develop. Will MERGE to develop but NOT deploy until the bank is materialized.
- PROTEA producer+scorer MERGED to develop (PR #733, auto-merged on green): pin bump to contracts
  21f300f + `_protst_text.py::apply_protst_text` (stamp-only, top-30 numpy kNN over bd3cd470 via
  DEFAULT_PROTST_CONFIG_ID/PROTEA_PROTST_CONFIG_ID, cosine-weight vote, per-query-max norm, module cache)
  + `ProtstTextScorer` (thin read) + D45 seam `_protst_default_fields()` (NaN not 0.0) + provenance +
  leakage guard (exclude query accession, refs from pre-cutoff t0). Feature OFF by default
  (compute_protst=False) so the develop merge is INERT until enabled -> extra safety layer over the bank barrier.
  TWO DEVIATIONS TO WATCH: (1) golden parquet fixtures regenerated (roll-forward for the ADR-D45
  pool-injected exclusion; _LEGACY_FIXTURE_SHA 775611822dd9->e9981e263df6, golden sha ->bcb5453c5a0e) -
  verify empirically when running the export. (2) no new typed GOPrediction column for protst - the
  offline A/B (parquet) works without it, but the platform run_cafa_evaluation re-read path (spec S7 §5.2)
  may need a typed column + migration; follow-up for that phase.
- STEP-7 CODE COMPLETE. Still pending (all gated on the extraction finishing): step 6 (materialize
  bd3cd470 via re-key), then deploy -> export (compute_protst=on) -> predict -> offline A/B -> (if gate)
  servable combiner retrain + platform 9-cell eval.

## Pre-dispatch checks resolved (for the step-7 run phase)

- Deployed contracts confirmed v1.4.0/v5 -> the v5->v6 bump is correct.
- Sealed test frame = `evaluation_set` **6e41eb5b-df02-4400-95c5-9cef6c9029ed** (window_role='test',
  annotation-set pair c905dffa -> 2394b9a1, created 2026-06-25 = the temporal v227->v230 eval). It is the
  ONLY test-role eval set. The table is keyed by (old_annotation_set_id, new_annotation_set_id) +
  window_role - there is NO query_set_id / name / display_name column. run_cafa_evaluation targets this
  evaluation_set + the per-category reranker ids.
- Still open (resolve at run time): does the bank bd3cd470 cover the acc27f47 candidate reference pool
  (check after materialization); frame-parity restrict_gt_to_predicted/softprop mapping.

## Step 6 DONE (2026-07-13 ~03:47)

- Offline extraction finished: 527424 sequences in 16.34h (log DONE line).
- Integrity: all 527424 rows embedding_dim=512, 0 nulls, values in ProtST range.
- RE-KEY committed: UPDATE 527424 rows 594701e0 -> bd3cd470; verified bd3cd470=527424, 594701e0=0.
- DROP: SQL config 594701e0 deleted (DELETE 1); only canonical protst config bd3cd470 remains.
- Coverage: bd3cd470 = 527424 seqs = the champion base pool (by construction). ProtST is now a
  first-class materialized signal under an API-born config.
- NEXT: deploy develop (#733 + contracts v1.5.0 pin) -> export compute_protst=on -> predict -> offline A/B.

## A/B run + enrichment-perf incident (2026-07-13 ~06:30)

- Arm A DONE (baseline, protst EXCLUDED) armA_9cell.json: nk-bpo 0.4607, LK-BPO 0.4901, PK-BPO 0.3433
  (mean_per_cell 0.5001; MF/CC cells stronger). The BP wall is the baseline to beat.
- eval frame fully enriched with protst_text (enriched_eval.parquet, 100% coverage).
- INCIDENT: the offline train enrichment (enrich_protst.py) re-queries embeddings from the DB PER CHUNK;
  the final chunk (v225-v230) hit a pathologically slow server-side query (57min+ active, DataFileRead -
  same DB-slowness class as the exports). NOT a deadlock; the client blocked on a slow query. Conductor
  INTERVENED: pg_cancel_backend(the slow query) + killed the stuck enrich PID 2944236; SendMessage the A/B
  agent to re-enrich the train IN MEMORY (load bank bd3cd470 once as numpy, cosine kNN top-30, vote t0 GO,
  per-query-max norm, leakage-exclude query accession - no per-chunk DB round-trips) and finish arm B.
- LESSON: offline feature enrichment must load the bank once into memory; never per-chunk DB embedding
  queries against sequence_embedding (it spans all configs -> full-scan-slow). Relates the export-slow
  pathology [[project_export_dataset_slow_minijobs_off_2026_07_11]].

## Authoritative promotion+eval (2026-07-13, author authorized "promocionar y evaluar", chose global=arm B)

- Deployed develop 49a2d55 (#734 compute_protst plumbed into export; /annotate unchanged - protst off by default).
- Export DISPATCHED: job f060f133-abb2-4f55-be60-fb279566a95c (export_research_dataset, queue protea.training,
  output_name protst-global-train227-test230) = the SEALED champion config {k30, train v160..v227, test v230,
  taxonomy/alignments/classifier/self_prior/association=true, embedding d8979601, ontology aeb07c36} +
  compute_protst=true. RUNNING + non-degenerate (615960 embeddings, ref_cache hit, real aspect refs, split 1/14).
- FAST-PATH REJECTED (both decisive): (1) stable_feature_cache BREAKS a classifier-on export - it emits a
  KNN-only candidate pool (no candidate-row union) via _export_stable_cache_runner, and the sealed 0.4063
  needs the native-0.391 classifier candidate-pool union; (2) compute_protst is NOT in compute_stable_cache_key
  (_stable_feature_cache.py) so a hit would restore protst-less parquets. So the authoritative run must be the
  FULL SERIAL compute = MANY HOURS (parasail alignments over millions of pairs across 14 snapshot pairs dominates).
- MINIJOBS not applicable: the parallel path (export_coordinator, export_minijobs/) does NOT compute protst at
  all; #734 wired compute_protst into the SERIAL export_research_dataset only. Routing via minijobs -> degenerate.
- DOWNSTREAM: on SUCCEEDED -> Dataset registered -> TrainRerankerAutoOperation trains fresh nk/lk/pk RerankerModels
  (replacing champion lk=83d758fe nk=94f0030e pk=66c5a25f) + evaluates on the test split -> then predict
  (compute_protst=on) + run_cafa_evaluation (evaluation_set 6e41eb5b) -> the served-pipeline 9-cell. The
  serve-flip (repoint /annotate to the protst models) is HELD for the eval number + author OK (evaluate first).
- DB DSN correction: port 5432 (not 5433); password in worktrees/protea-deploy/protea/config/system.yaml.

## PIVOT 2026-07-13: author flagged TWO gaps -> CANCELLED the export, cheap gate FIRST

Author (11:xx): "me molesta que no hayamos hecho k-WTA a estos embeddings... no era obvio?" +
"antes de export dataset, faltan comprobaciones... small run, o ver las metricas del KNN simple".
BOTH fair. We used RAW protst (512-d mean-pool + text projection) as the reranker feature; the CHAMPION
retrieval space is a LEARNED k-WTA head over Ankh-base -> protst never got the k-WTA treatment (an
untested asymmetry). And I jumped to a many-hour export without the cheap kNN gate.

ACTION: CANCELLED export f060f133 (API cancel 200 -> status CANCELLED; but serial parasail does NOT
honor mid-flight cancel, so SIGKILL worker 3311157 (was 478% CPU) + relaunched the protea.training
worker (PID 3330564) with a live sibling's exact env via `env -i` from /proc; queue was 0/0 unacked so
no requeue; GPU freed to 102 MiB; no stack restart, no outage). Confirmed job CANCELLED.

CHEAP GATE (agent a2113152722d13668, storage/regen_headline/protst_repr/): reuse the champion knn_confirm
harness (step3/l10std_train_eval.py, reproduces 0.2150 exactly) to score, apples-to-apples, protst
representations at kNN mean9 + per-cell (focus BP): champion(0.2150 ref) vs protst_raw vs protst_zscore
vs protst_kwta(champion recipe head on protst pool). Answers: (Q1) does learned k-WTA beat RAW protst at
kNN esp BP; (Q2) does protst beat the champion as a retrieval space on THIS harness (text_scorer claimed
raw-protst 0.2455 vs champion 0.2150 but on a different harness). DECISION drives which protst
representation feeds the producer (protst_text_score IS a kNN vote -> best kNN repr = best producer input)
BEFORE re-launching the authoritative export. Then a SMALL RUN (1-2 snapshot pairs) pre-flight before the
full 14-pair export.

## SPLIT PROTOCOL (author check 2026-07-13, verified): export=data, TRAIN script holds out v225-v227

Author asked "no deberian ser hasta 225 y 225-27 para validar". VERIFIED: the sealed champion ALREADY
does exactly this. The export emits ALL train pairs incl v225-v227 as tagged data (dataset
clean-learned-train227-test230 train_snapshot_pairs = v160-v165 .. v220-v225, v225-v227; eval v227-v230)
- 227 MUST be in train_versions so the 225->227 pair exists. The TRAIN/VALID split is applied at the
reranker TRAINING step, NOT the export: results/clean_227230/train_rerank_227230.py hardcodes
VAL_PAIR="v225-v227" (held-out for LightGBM early_stopping 50, num_boost 3000), trains on the earlier
pairs (v160-v165 .. v220-v225), tests on eval.parquet v227-v230. So: TRAIN <= v220-v225, VALID = v225-v227,
TEST = v227-v230 (D40 leakage-free). The lab universal_runner val_holdout_snapshot was null in the
transversal runs - that is a DIFFERENT path; the sealed path is the standalone train_rerank_227230.py.
ACTION FOR THE TRAINING STEP: train the protst rerankers with the SAME train_rerank_227230.py (identical
VAL_PAIR=v225-v227 + early-stopping) so the A/B vs sealed is clean. No change to the running export.

## THIRD SHUTDOWN 2026-07-14 ~14:45 - author powering off with 13/14 splits done

Export c5eae32c was at splits 0-12/14 DONE, split13 (v225-v227, the HEAVIEST slice) in its slow IO
data-load phase (~1h in, build_records not started). Author powers off -> loses only split13's partial
work (recomputes on resume); splits 0-12 markers + align_cache + cooccurrence reads persist on /home.
RESUME (same as SECOND BOOT): message is ACKED so NO auto-resume -> on reboot run `bash
scripts/cold-boot.sh` (brings up api+workers+frontend; ngrok+frontend may need the manual restarts noted
below if cold-boot doesn't cover them) then RE-DISPATCH the IDENTICAL payload (POST /jobs,
export_research_dataset, queue protea.training, sealed RECIPE + compute_protst=true + output_name
protst-global-train227-test230) -> resume-dir 7ff6ecc90b29 skips 0-12, resumes split13. Then: dataset ->
train_rerank_227230.py (VAL_PAIR=v225-v227) -> predict + run_cafa_evaluation (evaluation_set 6e41eb5b) ->
authoritative 9-cell. Cancel the orphan c5eae32c after re-dispatch for tidiness (optional).

## SECOND BOOT 2026-07-14 ~00:00 - needed cold-boot + RE-DISPATCH (NOT auto-resume)

Second poweroff/boot: this time the app layer did NOT auto-recover (api/workers/frontend all down; only
docker + watchdog + ngrok up). Ran `bash scripts/cold-boot.sh` (documented fast-path, reuses env, no
poetry install) -> api 200 + frontend 307 + public 307 + 11 workers. The export did NOT auto-resume this
time because the training queue was EMPTY (0/0/0): last session the worker had already ACKED the message
(removed from queue) and was processing in memory, so the 2nd poweroff lost it with nothing to redeliver.
So: cancelled the orphan 6f6f9399 (API cancel -> CANCELLED) and RE-DISPATCHED the identical sealed payload
-> NEW job **c5eae32c-53ee-4ed7-beaa-b94699cf7d0a** RUNNING, same resume-dir 7ff6ecc90b29, skips 0-5,
resumes from split6. LESSON: auto-resume only survives a boot if the msg was still UNACKED at crash; once
acked+in-memory, a boot needs a manual re-dispatch (same payload).

## REBOOT + AUTO-RESUME 2026-07-13 21:xx (verified behaviour)

Machine rebooted mid-split6. On boot: docker (postgres/rabbitmq/minio/monitoring, restart=unless-stopped)
+ API :8000 + all 11 workers came back automatically. THE EXPORT AUTO-RESUMED: RabbitMQ REDELIVERED the
unacked export message on the killed worker's channel close -> training worker re-acked job 6f6f9399 (same
id, same payload -> same resume-dir 7ff6ecc90b29) and is skipping splits 0-5 (done markers survived) +
recomputing split6 from scratch (the ~2h 58k-query giant slice, lost to the reboot). So NO manual
re-dispatch was needed. Two things did NOT auto-restart and were fixed by hand (surgically, WITHOUT
manage.sh start/stop which would kill the running export worker): (1) ngrok was down -> `setsid ngrok
start --all` (3 tunnels back: protea/lafa/mlflow); (2) frontend next-server was down (public 502) ->
`PORT=3000 HOSTNAME=0.0.0.0 <nvm v20.10 node> .next/standalone/server.js` from apps/web -> public 307.
KEY LESSON: the export survives reboots via AMQP redelivery + the resume dir; only the in-flight split is
lost. Do NOT manage.sh start/stop while an export runs (it force-kills next-server AND workers).

## RESUME NOTE 2026-07-13 (author shut the machine down mid-export)

Export job 6f6f9399 was interrupted at splits 0-5/14 DONE (split6 in-progress lost). The work is SAFE: the
export is RESUMABLE - staging dir storage/export_resume/protst-global-train227-test230-7ff6ecc90b29/ +
train_split{0..5}.done.json markers + align_cache.sqlite (3.7GB) all persist on /home. Verified in code:
_runner.py `_train_split_or_resume` -> `if self._resume.is_complete("train", i)` SKIPS done splits
(emits split_resumed); cleanup() only on success. TO RESUME after reboot: cold-boot the stack, then
RE-DISPATCH the IDENTICAL payload (POST /jobs, operation export_research_dataset, queue protea.training,
the sealed RECIPE.md payload + "compute_protst":true + "output_name":"protst-global-train227-test230").
Same payload -> same resume-dir hash 7ff6ecc90b29 -> skips splits 0-5 instantly, continues from split6.
The Dataset row is NOT inserted until success, so the name is NOT taken -> re-dispatch is allowed. Then
continue: dataset -> train_rerank_227230.py (VAL_PAIR=v225-v227) -> predict + run_cafa_evaluation
(evaluation_set 6e41eb5b) -> authoritative 9-cell. (Old job 6f6f9399 may sit orphaned RUNNING in DB - the
re-dispatch is a fresh job; optionally cancel the orphan for tidiness.)

## SPLIT-0 VALIDATION 2026-07-13: GO - #734 emits protst non-degenerate end-to-end

First split (v160->v165) closed (train_split0.done.json). All 3 protst cols present in nk/lk/pk shards
(79 cols): protst_text_score finite 0.23-0.35 (pk .345/lk .254/nk .231), spread min 0.02 med 0.10 max 1.0
16k-225k uniq; protst_vote_fraction 0-1 (31 uniq); protst_present binary. NON-DEGENERATE. #734 plumbing
confirmed on real exported data (the "small run" the author asked for). Finite frac < the A/B ~55% because
split0 is the OLDEST pair, all aspects (protst coverage lower on old proteins); signal present w/ real
spread. Export continues (13 more pairs). GO.

## GATE RESULT 2026-07-13 (protst_repr/REPR_RESULT.md): PASS on RAW protst; k-WTA loses

k-WTA on protst LOSES at kNN (d2048 -0.0074, d4096 -0.0043 mean9, loses on BP too); z-score +0.003 flat on
BP. RAW protst = best protst repr = 0.2455 mean9. => keep the deployed apply_protst_text kNN-vote over the
raw bank bd3cd470 unchanged; NO representation change; re-launch the export with raw protst. SIDE-FINDING
(strategic, escalate): raw protst BEATS the champion 0.2150 as a RETRIEVAL space by +0.0335, all 9 cells.
Next: SMALL-RUN pre-flight (validate #734 emits protst_text end-to-end) THEN the full authoritative export.

## Offline run status

PID recorded in `storage/regen_headline/protst_extract/PID.txt`; log
`extract.log`. Config `594701e0-8fbb-4571-9f69-3d227eeaa3be` (SQL-inserted,
to be superseded by the API config). Keep it running: it validates the
backend end-to-end and provides a full materialization now; no GPU is wasted
because the vectors are adoptable / cross-checkable.

## Honest prior

ProtST subsumes the champion's BP signal at kNN (combined ~= protst-alone);
open question whether it ADDS through the reranker to clear #1. An "8/9,
PK-BPO lifts but not #1" is a valid reportable outcome.

## FOURTH: split13 generic-plan pathology diagnosed + fixed (2026-07-15, author OK)

SYMPTOM: after the 3rd boot, split13 (v225-v227) ran ONE sequence_embedding SELECT for
>10h stuck in wait_event=DataFileRead, DB spilled 6273 GB to temp. Splits 0-12 had
completed fine.

ROOT CAUSE (not legit-slow, a degenerate plan): the export issues the per-split embedding
load as a PARAMETERIZED prepared statement ($1 config, $2 chunk, $3 annotation_set). After
enough executions Postgres switched to a GENERIC plan that ignores the actual parameter
values, mis-estimated the embedding_config_id selectivity (it selects ~1/8 of the 28GB /
9.5M-row sequence_embedding), and chose a HASH JOIN that materializes the whole table and
spills ~6TB to temp - instead of the Index Scan on uq_seq_embedding_seq_config_chunk. EXPLAIN
with a LITERAL annotation_set_id proves the good plan (Index Scan, cost 0.56..5.62, rows=1);
the heavy part is only a one-time parallel seq scan of protein_go_annotation (23GB), minutes.
Cold OS cache after the reboot + split13 being the largest split = the storm that only hit
this one split. reltuples were intact but last_analyze was NULL (stats-file reset).

FIX (author chose "Fix + re-dispatch"):
1. ALTER DATABASE protea SET plan_cache_mode = force_custom_plan;  (new connections only, no
   restart, reversible) - forces per-value custom plans so the config filter uses the index.
2. SIGKILL training worker (PID 14691) -> client disconnect cancelled the 10h query 494, temp
   reclaimed. Marked old job fef8b657 status=CANCELLED so it is not re-leased.
3. Relaunched the protea.training worker via /proc-env from a live sibling (14707) - no
   cold-boot, no manage.sh. Consumer re-registered (log: "Consumer started. queue=protea.training").
4. Re-dispatched the BYTE-IDENTICAL payload (pulled from the cancelled job row, not memory) via
   POST /jobs -> new job 84680eeb-ec1b-4832-9e9e-0dd80921d136. Same payload -> same resume-dir
   hash 7ff6ecc90b29 -> splits 0-12 skipped (markers untouched), resumed at split13.
VERIFY: after re-dispatch, NO long query - only short cycling queries (1s go_prediction /
build_records, healthy build phase). The 10h+6TB pathology is gone.

Monitoring now tracks job 84680eeb (not fef8b657). Expect split13 in minutes, then eval split
+ Dataset protst-global-train227-test230 registration -> the SUCCEEDED train->eval->notify flow.

## FIFTH: why THIS export is slow = protst bank reloaded per split (2026-07-15, root cause found)

Author asked "before was assumable, what changed?". This is the FIRST export with compute_protst=true
(#732 backend / #733 producer / #734 plumbing; deploy HEAD = 49a2d55 #734). The sealed champion export
had NO protst.

MECHANISM (code diff, not speculation):
- Ankh reference pool uses `_load_reference_pool_cached` (protea/core/disk_cache.py) -> DISK-cached,
  loaded ONCE, reused across all splits.
- The ProtST producer `_protst_text.py::_load_reference_bank` uses a PROCESS dict `_BANK_CACHE` keyed by
  `(config_id, t0_annotation_set_id)`. The t0 set CHANGES every split -> cache MISS every split ->
  it RELOADS all ~527k protst embeddings (527424 rows, ~1GB) from the 28GB sequence_embedding table via
  ORM `.all()` EVERY split, no disk cache. Plus a per-query numpy kNN vote over the 527k x 512 bank.
- On COLD cache after the reboot, each per-split reload is slow random IO. Combined with minijobs-off
  serial build + split13 being the biggest split = ~4h+ for split13 (prior splits, warm, ran 33-101min).
- Confirmed: ankh bank 527426 rows, protst bank 527424 rows -> protst effectively DOUBLES the heavy
  per-split bank-load work. The long query (pid 39613, `Protein JOIN SequenceEmbedding[config,chunk0]
  JOIN distinct(protein_go_annotation@t0)`) is exactly `_load_reference_bank`.

CORRECTNESS: unaffected (values are right; only speed). Does NOT justify killing the run.

FOLLOW-UP PR (offered to author; after this export finishes, do NOT touch mid-run): give protst the same
disk-cache treatment as ankh in `_protst_text.py::_load_reference_bank` - the EMBEDDING matrix for a given
config is identical across splits; only the t0 GO-term lookup (`_load_reference_go_terms`) changes. Cache
the (accessions, matrix) to disk keyed by config_id alone, re-derive ref_go per t0. Turns N reloads of
527k rows into 1. Brings future exports back to "assumable".

## SHUTDOWN 2026-07-15 ~14:47 (author tired, cutting)

STATE AT CUT: JOB 84680eeb (export_research_dataset protst-global-train227-test230) was RUNNING at
split13 (~4.5h build), splits 0-12 done on disk, NO split13 marker yet (build in RAM). force_custom_plan
is SET on the DB (persists). fef8b657 CANCELLED. Loop STOPPED.

IF SUSPENDED (not powered off): worker 759342 survives, export continues on wake. Just re-arm monitoring.

IF POWERED OFF (split13 build lost, redoes on resume):
  1. cold-boot: cd worktrees/protea-deploy && bash scripts/cold-boot.sh  (brings up api+workers; check ngrok/frontend)
  2. verify force_custom_plan still set: psql -c "SELECT setconfig FROM pg_db_role_setting s JOIN pg_database d ON d.oid=s.setdatabase WHERE d.datname='protea';" -> should show plan_cache_mode=force_custom_plan (it's a persistent ALTER DATABASE; only re-apply if missing).
  3. re-dispatch the IDENTICAL payload (pull from the cancelled job row or RECIPE-style; output_name protst-global-train227-test230) via POST /jobs {operation:export_research_dataset, queue_name:protea.training, payload:<same>} -> resumes at split13 (marker hash 7ff6ecc90b29 skips 0-12).
  4. OPTIONAL (recommended, makes split13+eval FAST): first apply the FIFTH-section fix - disk-cache the
     protst bank in _protst_text.py::_load_reference_bank so it is not reloaded per split. Big speedup for
     the re-run; correctness unchanged.
  5. When Dataset protst-global-train227-test230 registers -> train 3 protst rerankers (train_rerank_227230.py,
     VAL_PAIR=v225-v227, include the 3 protst_text cols) -> import-by-reference -> PREDICT (compute_protst=on)
     + run_cafa_evaluation (eval set 6e41eb5b) -> authoritative 9-cell vs sealed 0.4063.

## SIXTH: TRUE root cause (live audit 2026-07-15, PROVEN) - supersedes FIFTH's caching theory

Author asked to audit WHILE running. Profiled the worker process (no kill): worker 759342 was 60/60
samples in state S, 0.6% CPU, all threads sleeping, blocked in poll on the PG socket - 100% IDLE waiting
on the DB, while disk read ~460MB/s. So the bottleneck is a POSTGRES QUERY, not Python/kNN/caching.

The query = pid 39613, ONE SELECT running 4h19m in IO/DataFileRead (the protst reference-bank load). EXPLAIN
of the SAME query for the two configs (only embedding_config_id differs) is decisive:
- ANKH d8979601: Nested Loop + Index Scan uq_seq_embedding_seq_config_chunk, rows=1/lookup -> GOOD.
- PROTST bd3cd470: **Merge Join (rows=1)** with Index Scan ix_embedding_config_id estimating **rows=1 for the
  whole config** -> planner thinks bd3cd470 has ~1 embedding when it has 527,424 -> degenerate merge that
  scans/sorts the whole bank with terrible locality -> 4h+ in DataFileRead.

ROOT CAUSE: stale column statistics on `sequence_embedding.embedding_config_id` - the protst config
bd3cd470 is NEW (materialised this campaign) and was NOT in the histogram/MCV at the last ANALYZE, so the
planner used default selectivity (rows=1). Ankh (veteran config) was in the stats -> good plan.
force_custom_plan did NOT save it because the custom plan is built on the bad estimate. NOTE: this reframes
FIFTH - the per-split reload (no disk cache) is a real minor inefficiency, but the 4h was the BAD PLAN, not
the reload volume.

FIX APPLIED (0.56s, safe - ShareUpdateExclusive does not block the running AccessShare SELECT): `ANALYZE
sequence_embedding;`. Re-EXPLAIN protst now = Nested Loop rows=103476 (same good plan as ankh). Persistent.
Effect: the in-flight split13 query 39613 keeps its old bad plan until it ends, but the EVAL split's protst
bank load (and any resume/re-dispatch) now uses the good plan -> fast.

TAKEAWAY for the pipeline: after materialising a NEW embedding_config, ANALYZE sequence_embedding BEFORE any
export/predict that filters by it. Candidate for a post-materialise hook.

## SEVENTH: applied ANALYZE + re-dispatched on the good plan (2026-07-15 ~15:05, option B)

ANALYZE sequence_embedding applied (SIXTH) -> protst bank load now plans like ankh (index nested-loop).
Killed the stuck 4h bad-plan worker (759342 / query 39613), CANCELLED job 84680eeb, relaunched worker
(759687) via /proc-env, re-dispatched byte-identical payload -> NEW JOB 64aa4286-e26e-465e-a5e5-bfe6f88ae293,
RUNNING, resumed at split13 (13 done-markers for splits 0-12 intact). split13 now builds on the GOOD plan.
MONITOR THIS JOB: 64aa4286 (NOT 84680eeb / fef8b657, both CANCELLED). ANALYZE is persistent, so any future
resume is also fast.

## EIGHTH: RESUME AFTER POWER-OFF (2026-07-15 ~15:10) - single source of truth to come back

Author is POWERING OFF (not suspend). Job 64aa4286 (split13) will die; splits 0-12 preserved on disk.
PERSISTENT across the reboot: `ANALYZE sequence_embedding` stats (protst config now in stats) + `ALTER
DATABASE protea SET plan_cache_mode=force_custom_plan`. So the resumed export uses the GOOD plan (no 4h tax).

WHEN BACK, do exactly:
1. cold-boot: `cd ~/Thesis2/worktrees/protea-deploy && bash scripts/cold-boot.sh` ; then check ngrok +
   frontend are up (they often need a manual nudge - see project_cold_boot_procedure).
2. (sanity, optional) confirm the plan is still good: EXPLAIN the protst bank load for config
   bd3cd470 -> must be a Nested Loop + Index Scan (NOT Merge Join rows=1). If it regressed, re-run
   `ANALYZE sequence_embedding;` (0.6s). Also confirm `SELECT setconfig ... plan_cache_mode` still
   force_custom_plan.
3. re-dispatch the IDENTICAL payload (below) via POST /jobs {operation:export_research_dataset,
   queue_name:protea.training, payload:<below>} -> resumes at split13 (marker hash 7ff6ecc90b29 skips 0-12),
   now FAST (~1-1.5h split13 + ~1-1.5h eval split -> Dataset).
   payload: {"k":30,"output_name":"protst-global-train227-test230","test_versions":[230],
   "compute_protst":true,"train_versions":[160,165,170,175,180,185,190,195,200,205,211,215,220,225,227],
   "compute_taxonomy":true,"compute_alignments":true,"compute_classifier":true,"compute_self_prior":true,
   "compute_association":true,"embedding_config_id":"d8979601-ea59-4de1-9c16-21036ed67c36",
   "ontology_snapshot_id":"aeb07c36-17db-4b0b-a560-a68376998476"}
   JWT: mint HS256 from PROTEA_JWT_SECRET (in the worker environ, or scratchpad worker_env.txt while it lasts,
   or worktrees/protea-deploy/.env.local).
4. When Dataset protst-global-train227-test230 registers -> train 3 protst rerankers
   (results/clean_227230/train_rerank_227230.py, VAL_PAIR=v225-v227, INCLUDE the 3 protst_text cols) ->
   import-by-reference -> PREDICT (compute_protst=on) + run_cafa_evaluation (eval set 6e41eb5b) ->
   authoritative 9-cell vs sealed 0.4063 -> report + AB_RESULT.md + memory.

FOLLOW-UP BACKLOG (next session, from this saga): (a) ANALYZE sequence_embedding automatically after
materialising ANY new embedding_config (post-materialise hook) - would have prevented the whole 4h;
(b) disk-cache the protst reference bank in _protst_text.py (FIFTH) - minor, secondary to (a);
(c) decide whether the authoritative number even needs the full compute_protst re-export or can enrich the
sealed pool (SPLIT13_AUDIT.md step 1).

## NINTH: reboot recovery + re-dispatch on the good plan (2026-07-16 ~01:30)

Author powered off + came back. On resume: verified BOTH fixes PERSISTED across the reboot
(force_custom_plan still set; protst bank EXPLAIN = Nested Loop + Index Scan, NOT Merge Join rows=1).
Infra autostarted; the operator watchdog cron (agent-farm/scripts/cold-boot.sh --quiet, flock-guarded)
brought up api + 11 workers (do NOT fight it - wait for its flock to release). Cancelled the power-off
zombie job 64aa4286. Scratchpad was wiped by the reboot -> re-sourced the JWT secret from the live
worker's /proc/<pid>/environ and the payload from the cancelled job row. Re-dispatched byte-identical ->
NEW JOB c1e3240f-6dc8-4fb5-8e41-0a8ee6417c8c, RUNNING, resumed at split13 (13 markers intact). On the good
plan -> expect ~2-3h to Dataset (split13 ~1-1.5h + eval split ~1-1.5h), NOT days.
MONITOR THIS JOB: c1e3240f (fef8b657 / 84680eeb / 64aa4286 all CANCELLED).
