# L10-std champion regeneration — execution plan

Author directive (2026-07-11): "regeneramos mejor, justificamos y documentamos".
Goal representation: **Ankh-base layer 10, per-dim z-scored (fit on the reference pool),
then a learned k-WTA code (dict=2048, top_k=128, hard-neg objective)**. Build it THROUGH
PRODUCTION and benchmark it head-to-head against the faithful champion baseline
(`d8979601` over base `08234f06`, job `d0bc085e`, kept untouched as the control).

Read `BETTER_KWTA_DESIGN.md` first: the +0.0167 L10-std win is a clean signal INSIDE the
local extraction harness (`scale_train.log`) but that harness does NOT reproduce the served
champion (L48 local 0.1422 vs served 0.2150). The win is a HYPOTHESIS until L10-std runs end
to end through production. This plan makes that auditable.

---

## 0. Facts extracted from the codebase (cite before you change anything)

### 0.1 The k-WTA hard-neg training recipe (the champion's actual recipe)
Source: `protea-reranker-lab/src/protea_reranker_lab/encoder_ablation.py`
`_train_encoder` (L175-223) + `ArmSpec` (L46-54), and the deployed champion head meta
(`storage/learned_encoders/ankh_base_hardneg.pt`).

- **Head**: `nn.Linear(in_dim=768, dict_dim=2048)` (weight + bias) then `topk_real(z, top_k=128)`
  = keep the 128 largest-|value| entries per row, zero the rest (`encoder_ablation.py:106-112`).
- **Input to the head**: `l2n(base)` = L2-normalised base embedding (`encoder_ablation.py:183`).
  For L10-std the base is the z-scored L10 vector, THEN L2, THEN Linear
  (order: z-score -> L2 -> Linear -> top-k).
- **Objective `hard-neg`** (`encoder_ablation.py:187-198`):
  - random pairs: `sample_pairs(n_pool, train_pairs=300_000, rng(seed))`;
  - PLUS mined hard negatives: `anchors = rng.choice(n_pool, size=min(2000, n_pool))`; for each
    anchor take cosine top-`knn=30` neighbours in L2-normalised base space (self masked to -1),
    add each `(anchor, neighbour)` pair.
  - target `y = lin_pairwise(closures, IC, pairs, best_ic)` = Lin GO semantic similarity from the
    pool GO closures + information content (`encoder_ablation.py:199`, `sdr.lin_pairwise`).
- **Loss** (`encoder_ablation.py:206-215`): mean over pairs of `(cos(z_i, z_j) - y)^2`,
  minibatched `bs=32768`, accumulated `loss += ((cos - y)**2).sum() / n_pairs`.
- **Optim**: Adam, `lr=1e-3`, `epochs=150` (`encoder_ablation.py:203, 206`).
- **dtype** float32, device cuda.
- **Deployed champion head meta** (verified by loading the .pt):
  `{in_dim:768, dict_dim:2048, top_k:128, objective:"hard-neg",
    source_embedding_config_id:"08234f06-ba76-4d7d-aaec-ae601096b4fa",
    l2_normalize_input:true, band:"v227", reference_n:100000, seed:42}`, no `pooling` key
  (=> mean-pool family in `apply_learned_encoder._build_mean_apply`). state_dict = `weight (2048,768)`, `bias (2048,)`.

### 0.2 The z-score fit recipe (the lever)
Source: `storage/layer_ablation/scale_train.py:128-131` (`zscore_stats`):
```
mu = base.mean(0)            # per-dim mean over the training pool
sd = base.std(0) + 1e-6      # per-dim std over the training pool (+eps guard)
z  = (x - mu) / sd
```
Fit on the **training pool** (the 100k reference), applied to pool + queries + reference
(`scale_train.py:125-143`). Store as `scaler.npz` with keys `mu`, `sigma` (`sigma = sd`, already
includes the +1e-6). **Rigour fix vs the offline harness**: fit `mu`/`sigma` on the PRODUCTION
base-L10 embeddings (pulled from the DB), not the local extraction — this is exactly the
extraction-harness confound the whole exercise resolves.

### 0.3 The layer_indices for Ankh layer 10
Source: `protea-backends/src/protea_backends/ankh/__init__.py:301-313` (`_aggregate_layers`):
`hidden_states[-(li + 1)]` for each `li` in `layer_indices`.
Verified: `storage/layer_ablation/emb_ankh_base/meta.json` -> `nlayers=49, dim=768`
(embeddings + 48 encoder layers; `extract_layers.py` and `scale_extract.py` index `hidden_states[L]`
directly, L in {10,19,48}).
- Champion base L48 = `layer_indices=[0]` -> `hidden_states[-1]` = `hidden_states[48]` = last layer.
- **L10 = `layer_indices=[38]`** -> `hidden_states[-39]` = `hidden_states[49-39]` = `hidden_states[10]`.
- General rule: `li = 48 - target_layer`.
- **Assert at compute time** `len(hidden_states) == 49`, else recompute `li = (nlayers-1) - target_layer`.

### 0.4 How a learned-code head + config are loaded/applied in production
Source: `protea/core/operations/apply_learned_encoder.py` + `_learned_code_embed.py` +
`compute_embeddings.py:387-458`.
- `apply_learned_encoder` (op) APPLIES a pre-trained head over every source embedding, creating a
  new `EmbeddingConfig(model_backend="learned-code")` idempotently
  (`_ensure_target_config` L161-183), named
  `learned-code:{pool_tag}:{objective}:{source_id[:8]}` (mean-pool => `pool_tag="mean"`), and stores
  the 2048-d codes as ordinary `SequenceEmbedding` rows. The unchanged predict/KNN/reranker/cafaeval
  pipeline then consumes them by pointing at the new `embedding_config_id`.
- On the `/annotate` serve path, `_learned_code_embed.embed_learned_code` resolves the base config by
  the id-prefix in the learned config name (`resolve_base_config` L91-126), resolves the head `.pt`
  from `PROTEA_LEARNED_ENCODER_ARTIFACT` (explicit) or `PROTEA_LEARNED_ENCODER_DIR/<config_id>.pt`
  (then `<config_id[:8]>.pt`) (`resolve_encoder_artifact` L129-155), and applies it via `_load_encoder`.
- **NO z-score exists in production today** — confirmed: `_build_mean_apply` did mean-pool -> L2 ->
  Linear -> top-k with no standardisation. This PR (#730, branch `feat/learned-code-zscore-scaler`,
  DRAFT) adds the optional scaler-travels-with-the-head piece (see section 4).

### 0.5 Where the k-WTA training runs
There is **no platform operation that TRAINS a learned-code head**; `apply_learned_encoder` only
APPLIES one, and encoder/LightGBM training was deliberately decoupled into `protea-reranker-lab`
(see PROTEA `CLAUDE.md` "Re-ranker training decoupling"). => **The k-WTA hard-neg retraining is a
documented OFFLINE run** in `protea-reranker-lab` (MLflow-tracked), reading PRODUCTION base-L10
embeddings from the DB (`encoder_ablation.pull_mean`). Everything downstream is platform-dispatch.

---

## 1. Reproducibility anchors (pin these; the result must be auditable, not debugged)

| Anchor | Value | Why |
|---|---|---|
| Seed | `42` (head training + hard-neg mining rng) | matches the champion head meta |
| Pool | the SAME 100k v227-annotated reference pool as the champion (`band=v227`, `reference_n=100000`) | so the ONLY base difference is the layer |
| Pool accessions | pin & save `pool_accs.json` (exact accession list) | deterministic re-run; leakage asserts |
| Snapshot pair | TEST v227->v230 (6mo), VALID v225->v227 | the sealed 0.4063 frame; the running baseline `d0bc085e` |
| z-score fit set | the 100k pool, on PRODUCTION base-L10 embeddings | removes the extraction-harness confound |
| Base config | identical to `08234f06` EXCEPT `layer_indices=[38]` | isolate the layer as the only base change |
| dtype | float32 (z-score fit + head training) | mid-layers overflow fp16 (`|max|` up to ~4.9e5) |
| Head meta | `{in_dim:768, dict_dim:2048, top_k:128, objective:"hard-neg", seed:42, reference_n:100000, band:"v227", source_embedding_config_id:<L10 base id>, l2_normalize_input:true}` | provenance in the .pt |
| Control | job `d0bc085e` (champion `d8979601`) LEFT RUNNING & UNTOUCHED | honest head-to-head baseline |

Leakage asserts (from `scale_train.py:96-97`): pool ∩ queries == ∅ and pool ∩ reference == ∅.

---

## Author add-on (2026-07-12): full-data head variant at Step 3 (cheap - do it here)
The k-WTA HEAD training is fast (Linear 768->2048, ~1.5M params, 150ep on pooled vectors = minutes; the
EXPENSIVE part is compute_embeddings, running now). Step 2 materialises ALL ~527k L10 embeddings, so once
it finishes we have the full data for free. At Step 3, train the head on BOTH:
  (a) the SAME 100k v227 pool as the champion  -> the PRIMARY apples-to-apples arm (L10-std vs L48 champion);
  (b) the FULL ~527k reference pool             -> a CHEAP bonus arm testing "does full training data help on
      the PRODUCTION base?" (the pool-size 15k->100k null was on the WEAK local base + confounded by ref_n
      setting both pool AND reference; a clean test = vary the TRAINING pool while holding the KNN
      reference/eval FIXED).
Compare (a) vs (b) on the same 9-cell head-to-head. Keep (a) as the headline apples comparison; (b) is a
low-cost extra data point.
PLUS (author-approved 2026-07-12) a CHEAP KNN-level SCREEN of the head hyperparameters (never swept):
top_k in {64,128,256} x dict_dim in {2048,4096} (~6 variants). Train each head (minutes) + evaluate at
the KNN-retrieval mean9 level ONLY (cheap, ablation-harness style, on the production L10 base) - NOT the
full pipeline per variant. Only the screen WINNER (if it beats the champion's 2048/128) earns a full
export->reranker->predict->cafaeval run. Honest prior: LOW yield (pool-size + hard-neg were null; the
champion's edge is the base not the geometry) but top_k was never touched and the screen is cheap. (Champion clarification: served champion d8979601 = LAST layer 48; L10-std = the
MIDDLE-layer-10 + z-score CHALLENGER being tested through production - not an established result.)

## Progress log
- 2026-07-11 Step 1 DONE: registered L10 base config `L10_BASE_ID = 81436dba-1324-4536-bccc-122ac45dd9ba`
  (ankh-base-L10, layer_indices=[38]); diff vs champion base 08234f06 confirmed = ONLY layer_indices
  differs ([38] vs [0]), all else identical. Admin bearer sufficed for POST /embeddings/configs.
- 2026-07-12 Step 2 DONE: compute_embeddings L10 (job 076bfe6d, embedding_scale=32) SUCCEEDED = 527,861
  SequenceEmbedding rows for 81436dba (the full production L10 base, stored as L10/32; z-score absorbs it).
  Halfvec-overflow fixed by #731 (scale=32, 0 clips). ~9h GPU run.
- 2026-07-12 Step 3 IN FLIGHT: offline lab agent training the k-WTA heads (primary 100k apples + full-data
  527k + top_k/dict_dim KNN screen), KNN-level mean9, vs the L48 champion baseline. Results ->
  storage/regen_headline/L10STD_STEP3_RESULTS.md + winner head in step3/.

## 2. Execution steps

### Step 1 — Register the L10 base EmbeddingConfig  [PLATFORM: POST /embeddings/configs]
Create a config identical to champion base `08234f06` except the layer.
`POST /embeddings/configs` (`create_embedding_config`, operator role) with:
```json
{
  "model_name": "ElnaggarLab/ankh-base",
  "model_backend": "ankh",
  "layer_indices": [38],
  "layer_agg": "mean",
  "pooling": "mean",
  "normalize": false,
  "normalize_residues": false,
  "use_chunking": false,
  "max_length": <same as 08234f06>,
  "display_name": "ankh-base-L10",
  "description": "Ankh-base layer 10 (hidden_states[10]=layer_indices[38]); L10-std champion base"
}
```
Record the new config id (call it `L10_BASE_ID`). Diff it against `08234f06` and confirm the ONLY
difference is `layer_indices` ([38] vs [0]).

### Step 2 — Materialise base-L10 embeddings for the pool + queries  [PLATFORM: POST /jobs]
`POST /jobs {operation:"compute_embeddings", payload:{embedding_config_id: L10_BASE_ID, <accession/sequence scope for the 100k pool + the eval queries>, device:"cuda"}}` (queue `protea.embeddings`).
At compute time assert `len(hidden_states)==49` (guards the layer arithmetic). This produces the
PRODUCTION base-L10 vectors the head + scaler are fit on.

### Step 3 — [OFFLINE, protea-reranker-lab, MLflow] Fit z-score + train k-WTA hard-neg head
Reuse the recipe verbatim (section 0.1/0.2). Concretely, an MLflow-tracked lab script that:
1. `pull_mean` the pool's base-L10 embeddings from the DB for `L10_BASE_ID` (the SAME 100k pool accs);
2. fit `mu = X.mean(0)`, `sigma = X.std(0) + 1e-6` on the pool -> save `head.scaler.npz` (`mu`, `sigma`);
3. z-score the pool base, then run `_train_encoder`-equivalent with
   `ArmSpec(kind="learned", dict_dim=2048, top_k=128, objective="hard-neg")`, `seed=42`,
   `epochs=150`, `train_pairs=300_000`, `knn=30`, `lr=1e-3`, `bs=32768`, float32;
4. save `head.pt = {"state_dict": Linear.state_dict(),
   "meta": {in_dim:768, dict_dim:2048, top_k:128, objective:"hard-neg", seed:42, reference_n:100000,
   band:"v227", source_embedding_config_id: L10_BASE_ID, l2_normalize_input:true}}`
   (NO `pooling` key => mean-pool family), with `head.scaler.npz` as its **sibling** (same stem).

**Parity check (resolves the confound):** for a sample of shared accessions, compare production
base-L10 vectors (step 2) against `storage/layer_ablation/scale_pool_emb/layer_10.npy`. Large drift
means the extraction harness != production (the exact defect that made local L48=0.1422 vs served
0.2150). Log the parity delta as an MLflow artifact.

### Step 4 — Apply the head -> learned-code codes for the pool  [PLATFORM: POST /jobs]
Place `head.pt` + `head.scaler.npz` together (same stem) on a path the worker can read.
`POST /jobs {operation:"apply_learned_encoder", payload:{source_embedding_config_id: L10_BASE_ID,
encoder_artifact_path: "<...>/head.pt", target_model_name:"learned-code"}}` (queue `protea.jobs`).
- The scaler is auto-discovered as the `.pt` sibling (PR #730) and z-scores the base BEFORE the
  in-head L2 + Linear — reproducing L10-std end to end.
- This creates the learned-code config `learned-code:mean:hard-neg:<L10_BASE_ID[:8]>` (call it
  `L10STD_ID`) and stores the codes. Record `L10STD_ID`.
- For the later serve path, copy/symlink the head + scaler into `PROTEA_LEARNED_ENCODER_DIR` as
  `<L10STD_ID>.pt` + `<L10STD_ID>.scaler.npz` (config_id -> `<config_id>.pt` then `<config_id[:8]>.pt`).

### Step 5 — Export the research dataset on the new config  [PLATFORM: POST /datasets]
`POST /datasets` (enqueues `export_research_dataset` on `protea.training`) with
`embedding_config_id = L10STD_ID`, the snapshot pair (TRAIN v227->…, EVAL v227->v230),
`k`, `annotation_source` matching the champion export. Yields `train.parquet` / `eval.parquet` /
`manifest.json` + a `Dataset` row.

### Step 6 — Train the reranker  [OFFLINE, protea-reranker-lab] then register  [PLATFORM]
Existing decoupled flow: `pull_dataset.py` -> LightGBM -> `POST /reranker-models/import`
(validates `feature_schema_sha`, links `dataset_id`). Keep the reranker spec identical to the
champion's so the only moving part is the embedding representation.

### Step 7 — Predict  [PLATFORM: POST /jobs]
`POST /jobs {operation:"predict_go_terms", payload:{embedding_config_id: L10STD_ID,
reranker_model_id: <new>, <snapshot pair / query scope>}}` (queue `protea.predictions`).

### Step 8 — Evaluate  [PLATFORM: POST /jobs]
`POST /jobs {operation:"run_cafa_evaluation", payload:{prediction_set_id: <from step 7>, …}}`
(queue `protea.evaluations`). Produces the 9-cell f_micro_w.

### Step 9 — Benchmark head-to-head + stratify  [analysis]
Compare L10STD vs the faithful champion baseline (`d0bc085e`) on the ONE metric (f_micro_w) and the
ONE frame (v227->v230), STRATIFIED over the 9 cells (NK/LK/PK x MFO/BPO/CCO) plus length x
neighbour-identity, with bootstrap CIs on the per-cell deltas and Wilcoxon+Holm across cells
(the standing 3-axis norm). This resolves the confound: is L10-std actually better THROUGH production?

### Step 10 — Decide + document  [thesis + UI + Sphinx]
- If L10-std beats the champion through production: promote as the new SOTA representation; write an
  ADR + WRITEUP; the regenerated headline uses it.
- If it does NOT: document that the debugging-era L48 champion was near-optimal through production
  (the confound explained) and keep it.
Either outcome is a rigorous, publishable finding. Surface the claim on all three surfaces
(thesis, interface, Sphinx), not just a log.

---

## 3. Platform-dispatch vs offline-documented (summary)

| Step | Where |
|---|---|
| 1 Register L10 base config | PLATFORM `POST /embeddings/configs` |
| 2 compute_embeddings (base L10) | PLATFORM `POST /jobs` (protea.embeddings) |
| 3 z-score fit + k-WTA hard-neg training | **OFFLINE** protea-reranker-lab, MLflow (no platform train op) |
| 4 apply_learned_encoder (codes) | PLATFORM `POST /jobs` (protea.jobs) |
| 5 export_research_dataset | PLATFORM `POST /datasets` (protea.training) |
| 6 reranker train / import | OFFLINE lab + PLATFORM `POST /reranker-models/import` |
| 7 predict_go_terms | PLATFORM `POST /jobs` (protea.predictions) |
| 8 run_cafa_evaluation | PLATFORM `POST /jobs` (protea.evaluations) |
| 9 benchmark + stratify | analysis |
| 10 decide + document | thesis/UI/Sphinx |

**Can the k-WTA retraining be platform-native?** Not today. No operation trains a learned-code head;
training is decoupled to `protea-reranker-lab` by architecture. It stays a documented offline run
that reads PRODUCTION embeddings and is MLflow-tracked. (A future `train_learned_encoder` platform op
could close this, but it is out of scope for this regeneration.)

---

## 4. The production-code piece (implemented)

PR **#730** (branch `feat/learned-code-zscore-scaler`, base `develop`, DRAFT = held for author review):
optional per-dim z-score scaler travels with the head. When a learned head has a sibling
`<stem>.scaler.npz` (`mu`, `sigma`, each `(in_dim,)`), the base is standardised `(x - mu)/sigma`
BEFORE the in-head L2 + Linear, in `apply_learned_encoder._load_encoder` (shared by BOTH the offline
`apply_learned_encoder` op and the `/annotate` serve path). Resolution: `PROTEA_LEARNED_ENCODER_SCALER`
(explicit) -> `PROTEA_LEARNED_ENCODER_DIR/<config_id>.scaler.npz` (or `[:8]`) -> head sibling.
Backward compatible: no scaler => byte-for-byte the previous behaviour, so the existing champion is
untouched. Malformed scaler => clear ValueError. Unit tests added; ruff/mypy/smell/pytest green.

---

## 5. Do-not-touch

- Do NOT stop or perturb the running faithful baseline export `d0bc085e` (it IS the control).
- Do NOT run any GPU job / dispatch compute_embeddings/training as part of THIS scoping task.
- The sealed 0.4063 is immutable; regenerated numbers are CANDIDATES until the head-to-head lands.
