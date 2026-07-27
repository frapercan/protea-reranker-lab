# Native TEST seal kit (H-azucar boosters, le227)

Prepared autonomously 2026-06-20. **Nothing here has been fired.** These are
the artefacts + payloads to obtain the native LAFA TEST f_micro_w on
`/benchmark` (7401 targets, frame 227 to 230), pending the user's go.

The 3 boosters in `../ensemble_gbm_{NK,LK,PK}.txt` were refit on the
combined train+eval rows with t0 at or before v227 (H-azucar), reusing the
validated early-stop rounds (NK 542 / LK 170 / PK 769). They were trained on
the **69-feature** schema of dataset `fullgo-union-SELECT-160-220-227-v5`,
which **includes the v6 block** (emb_pca x16, anc2vec, taxonomy_voters,
go_context/lineage).

## The corrected path (supersedes the earlier v6-off plan)

The boosters were trained WITH the v6 features (real values), and the offline
0.391 champion they replicate uses them. Therefore reproducing the champion
**requires v6 ON at serve**:

| Path | serve flags | sha | boosters get v6 as | verdict |
|------|-------------|-----|--------------------|---------|
| **A (recommended)** | v6=TRUE | `94e87ae6f4ed` | real values (match train) | faithful, best shot at 0.391-class |
| B (v6-off) | v6=FALSE | `7fcecf26aa0a` | NaN (~44 features dead) | train/serve skew, degraded - AVOID |

`infer_active_feature_families` DOES know the v6 families, and the anc2vec npz
is present in the deploy (`PROTEA_ANC2VEC_PATH`), and emb_pca is fitted
transductively over the full pool at serve (mirrors the offline dump). So
Path A is viable. This **reverses** the prior memory note that read
"v6=True -> HARD-FAIL": v6=True only fails if the boosters are sealed under the
v6-off sha. Seal them under `94e87ae6f4ed` and v6=TRUE matches cleanly.

The spec.yaml / run.json here seal under **`94e87ae6f4ed`** (Path A).

## HARD PREREQUISITE: build cooccurrence for v227

The association feature (PK #1, ~22% gain) reads `term_cooccurrence` keyed by
the **t0 reference annotation_set**. The TEST frame's t0 set is
**v227 = `c905dffa-a5ce-430b-b17b-503e88666adb`** (verified: PredictionSet
12739db3 uses exactly this set + snapshot 35c3ad67). Cooccurrence is currently
built only for **v160..v220** (the 13 training sets); **v227 is MISSING**.

Without it the association feature is all-zero at TEST serve -> the PK#1 lever
dies -> the same collapse that crippled the first native attempt (0.315).

Fix: dispatch `build_go_cooccurrence` for v227 BEFORE the predict
(payload in `prereq_cooccurrence_v227.json`). v227 has 599k experimental rows /
88k proteins, same magnitude as v220 (581k / 85.9k -> 25.2M cooccurrence rows),
so the bounded COPY-path build is the same cost as the 13 already done.
Idempotent (delete+rewrite). Run it when RAM is free (the lever holds ~36G).

## Order of operations (all gated on user OK)

1. `POST /jobs` build_go_cooccurrence  (prereq_cooccurrence_v227.json)
2. `POST /reranker-models/import-by-reference` x3 (NK/LK/PK; spec+run here)
3. `POST /jobs` predict  (predict_payload.json) -> new PredictionSet on 7401
4. `POST /jobs` run_cafa_evaluation  (eval_payload.json) -> EvaluationResult
5. read f_micro_w from /benchmark

## Remaining gap to exactly 0.391 (identified, not unknown)

- **TOI**: run_cafa_evaluation loads the whole snapshot ontology as terms of
  interest, not the official ~38.6k frame-delta TOI. No code path injects an
  external TOI -> closing it is a CODE change. The number will differ in
  magnitude until then. Baseline KNN-only EvaluationResult = `37a93417`.
