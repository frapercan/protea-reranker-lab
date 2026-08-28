# Feature leakage ruling: universal multi-PLM pooling columns (F-RERANK-UNIVERSAL.2)

Companion to the base audit in `protea-deploy/docs/FEATURE_LEAKAGE_AUDIT.md`
(F-EVAL-PROTOCOL.b). This document records the explicit GO/NO-GO ruling for
columns introduced in the F-RERANK-UNIVERSAL.2 slice: the 4 latent
`lineage_*` columns (present in parquet, absent from ALL_FEATURES before v3),
and the two pool-stage-injected columns `plm_id` and `k_context`.

## Where this ruling is enforced (updated with contracts v1.7.0)

The `lineage_*` ruling below was originally enforced by the contracts
catalogue itself: the columns were in the parquet but not in `ALL_FEATURES`,
so nothing trained on them. Contracts v1.7.0 (v6 schema) put them back into
`ALL_FEATURES`, which is a catalogue decision on the producer side and does
not overturn this ruling.

The ruling is now enforced in the lab, where it belongs. The default training
set is `protea_reranker_lab.contracts.DEFAULT_TRAINING_FEATURES`, the
catalogue minus `UNADOPTED_FEATURE_FAMILIES`, which holds out `lineage`. The
guard lives in `tests/test_feature_count_guard.py`. The NO-GO ruling below is
unchanged, and so is the set of columns actually trained on: the default set
still hashes to schema sha `a0986dedd912`.

Re-evaluation still requires a controlled ablation. The family stays
reachable through `enabled_feature_families=["lineage", ...]`, which is how
that ablation should switch it on.

## Golden rule (from base audit)

> A feature must be computable **identically** for a never-seen protein with
> **zero known labels**, using only data dated at or before the training
> cutoff (t0).

Additionally checked here: the **bucket-id shortcut** anti-pattern. A
column must not serve as a proxy identifier for the evaluation split,
category membership, or label density, even if its definition is
temporally honest. The canonical cautionary case is
`anc2vec_query_known_count`, which was a near-perfect category-id due to
the shard replication artifact (PROTEA fix 223299c). See the base audit
for the detailed diagnosis and fix.

## check_cutoff_guard applicability

The `check_cutoff_guard` logic in `band_registry_bridge.py` enforces
temporal honesty (no future-data artifacts) at the band level. It is
orthogonal to this audit: it guards the *evaluation window*, not the
*feature construction*. Both checks must pass independently.

Applied here: `plm_id` and `k_context` carry no date at all and are
injected from the manifest's declared K and embedding_config_id, which
are derived from training-time decisions, not from future annotations.
The cutoff guard does not fire on them. `lineage_*` columns are computed
using the query's pre-cutoff known set (same as `anc2vec_query_*`), which
passes the temporal guard.


## Column-by-column ruling

### lineage_is_ancestor_of_known

**Definition**: 1 if the candidate GO term is a (direct or transitive)
ancestor of ANY term the query protein was annotated with before t0. 0
otherwise. Source: `protea_method.lineage`, input = `eval_data.known`.

**Temporal honesty**: YES. Uses only pre-cutoff known annotations.

**Empty-protein handling**: For a protein with zero pre-cutoff known terms,
`eval_data.known` is empty and all four lineage columns are 0
(the producer's documented default in `_lineage_default_fields`).
Output is identical to the genuine-NK case.

**Bucket-id risk**: Moderate. The value is 1 for ancestors of the protein's
known terms and 0 otherwise. If the shard replication artifact were still
present, this column would partially encode category membership (NK proteins
have empty known, LK/PK have non-empty). The fix in PROTEA 223299c
(category-disjoint shard construction) eliminates this risk: each
(protein, aspect) appears in exactly one category shard, so
`lineage_is_ancestor_of_known` varies for the genuine reason (how many
pre-cutoff ancestors the protein has) rather than as a category boundary.
The same structural argument applies as to `anc2vec_query_*` in the base
audit.

**RULING: NO-GO (DEFAULT EXCLUDE)**

Reason for default exclusion: the base audit already admits `lineage_*`
columns are in the parquet but excluded from `ALL_FEATURES`. The decision
to keep them out of the canonical training set is preserved here. The
temporal and construction correctness arguments above are valid, but the
marginal risk of any residual bucket-id effect from the shard construction
does not justify inclusion without a controlled ablation showing positive
Fmax delta. The universal pooling context is already a new source of
complexity; the leakage surface should not be expanded speculatively.
Re-evaluate in a dedicated slice (F-RERANK-UNIVERSAL.X) after the
baseline pooled run is established.

### lineage_is_descendant_of_known

**Definition**: 1 if the candidate GO term is a (direct or transitive)
descendant of ANY term the query had before t0. 0 otherwise.

**Temporal honesty**: YES. Same inputs as above.

**Empty-protein handling**: 0 for empty-known proteins.

**Bucket-id risk**: Same analysis as `lineage_is_ancestor_of_known`.

**RULING: NO-GO (DEFAULT EXCLUDE)** Same reasoning as above.

### lineage_ancestor_of_count

**Definition**: Number of pre-cutoff known GO terms of which the candidate
is an ancestor.

**Temporal honesty**: YES. Count over pre-cutoff known terms only.

**Empty-protein handling**: 0 for empty-known proteins.

**Bucket-id risk**: Moderate. For NK proteins (empty known) this is always
0. For LK/PK proteins it varies. If category shards were not disjoint, this
would be a near-perfect NK-vs-{LK,PK} discriminator, directly analogous to
the `anc2vec_query_known_count` replication artifact. The structural fix
applies. However, `anc2vec_query_known_count` was specifically identified
as "the offender" in the historical incident and was verified fixed; the
`lineage_*` columns received the same structural fix but have not undergone
the same post-fix ablation in the leakage experiments. Prudence requires
excluding them by default.

**RULING: NO-GO (DEFAULT EXCLUDE)**

### lineage_descendant_of_count

**Definition**: Number of pre-cutoff known GO terms of which the candidate
is a descendant.

**Temporal honesty**: YES. Same inputs.

**Empty-protein handling**: 0 for empty-known proteins.

**Bucket-id risk**: Same as `lineage_ancestor_of_count`.

**RULING: NO-GO (DEFAULT EXCLUDE)**


### plm_id (pool-stage injected, v3+)

**Definition**: String identifier of the protein language model used to
produce embeddings for KNN retrieval (e.g. `"prot_t5"`, `"esm2_650m"`).
Injected at pool-stage time as a constant per manifest source; absent from
the raw parquet dumps.

**Temporal honesty**: YES. The PLM choice is a training-time infrastructure
decision; it does not depend on future annotations or the evaluation window.

**Empty-protein handling**: Injected unconditionally from the manifest
metadata; value is defined even for a never-seen protein.

**Bucket-id risk: HIGH (conditional GO)**

This is the critical risk case, directly analogous to the
`anc2vec_query_known_count` replication artifact. The concern:

- The 24 v226-lineage manifests are partitioned by (PLM, K). If a protein
  appears in only one PLM's manifests (because different PLMs cover different
  protein subsets due to embedding availability or retrieval gaps), then
  `plm_id` would deterministically identify that protein's subset and could
  act as a label-proxy.
- However, in the actual v226-lineage dataset construction ALL 8 PLMs are
  applied to the SAME protein pool (the full GOA v226 SwissProt corpus).
  Every protein appears in every PLM manifest, so `plm_id` does not partition
  the protein set. It varies within a protein's rows (same protein, same GO
  candidate, different PLM retrieval contexts) and thus cannot be a
  protein-subset indicator.
- The residual risk is that a PLM's retrieval quality is correlated with
  label density in a way that makes `plm_id` a proxy for labelling ease.
  This is a real but moderate risk, mitigated by: (1) all PLMs train on the
  same label set, (2) the booster sees all PLMs simultaneously in the pooled
  view, so it cannot overfit to a single PLM's label density, and (3) the
  canonical ablation (`plm_id` ablation in F-RERANK-UNIVERSAL.5) is the
  correct post-hoc guard.

**Condition for GO**: `plm_id` is admitted as a categorical feature under
the following structural guarantee, which must be verified in the smoke pool:
(a) every protein appears in ALL PLM manifests (no protein-subset partition),
(b) within a (protein, go_term) pair, rows from different PLMs may differ in
the KNN-derived columns, so `plm_id` encodes genuine retrieval-context
variation rather than a split label, and (c) the F-RERANK-UNIVERSAL.5
ablation run must confirm that removing `plm_id` does not improve VALID
f_micro_w by more than 0.002 (a threshold above which collapse to the
strongest PLM would be evident).

**RULING: CONDITIONAL GO**

`plm_id` is included in `ALL_FEATURES` (v3) and in the `plm_context` family.
It will be injected at pool-stage time. The ablation guard (condition c) must
be executed in F-RERANK-UNIVERSAL.5 and its result appended to this
document. If the ablation shows collapse, `plm_id` is dropped and the v3
schema is revised.


### k_context (pool-stage injected, v3+)

**Definition**: Integer KNN neighbourhood size (3, 5, or 10) used to
retrieve the candidate from the embedding index. Injected at pool-stage
time as a constant per manifest source; absent from the raw parquet dumps.

**Temporal honesty**: YES. K is a retrieval hyperparameter fixed at training
time; it does not depend on future annotations.

**Empty-protein handling**: Injected unconditionally from manifest metadata.

**Bucket-id risk: LOW**

- K does not identify proteins; within a protein's pooled rows, K simply
  distinguishes which neighbourhood size produced the candidate row.
- K is directly informative about retrieval confidence: a candidate that
  appears under K=3 is a closer neighbour than one appearing only under K=10.
  This is a legitimate, temporally honest signal.
- The risk of K acting as a label proxy is minimal: K is a retrieval
  decision that precedes any label assignment and is the same for all
  proteins in the same manifest.

**RULING: UNCONDITIONAL GO**

`k_context` is included in `ALL_FEATURES` (v3) and in the `k_neighborhood`
family. No ablation condition is imposed.


## Summary

Columns and their rulings:

- `lineage_is_ancestor_of_known`: NO-GO (DEFAULT EXCLUDE). Temporally honest; structural fix verified; excluded pending ablation.
- `lineage_is_descendant_of_known`: NO-GO (DEFAULT EXCLUDE). Same as above.
- `lineage_ancestor_of_count`: NO-GO (DEFAULT EXCLUDE). Analogous to `anc2vec_query_known_count`; excluded pending ablation.
- `lineage_descendant_of_count`: NO-GO (DEFAULT EXCLUDE). Same as above.
- `plm_id`: CONDITIONAL GO. In ALL_FEATURES v3; ablation required in F-RERANK-UNIVERSAL.5.
- `k_context`: UNCONDITIONAL GO. In ALL_FEATURES v3; no ablation condition.


## lineage_* re-evaluation criteria (for future slice)

To clear `lineage_*` from DEFAULT EXCLUDE, a dedicated experiment must:

1. Train two models on the same pooled set: one with `lineage_*` in the
   active families, one without.
2. Compare VALID f_micro_w on the 226->227 band. If delta > 0.003 (above
   measurement noise at this resolution), proceed.
3. Run the same post-fix constructional verification as was done for
   `anc2vec_query_*` in the base audit: confirm that within the category-
   disjoint shards, `lineage_ancestor_of_count` varies continuously for the
   genuine reason (annotation density) rather than as a binary NK=0/LK-PK>0
   separator. A histogram of the column's per-protein mean value, stratified
   by category, is sufficient.
4. If both conditions pass, update this document and bump to a minor version.
