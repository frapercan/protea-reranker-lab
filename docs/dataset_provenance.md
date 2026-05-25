# Dataset Provenance and FAIR Compliance

This document covers the full data lineage, pipeline version, split
methodology, and FAIR compliance posture for the `bench-v1` dataset
family produced by PROTEA's `export_research_dataset` operation.

Slice: F-DATA-PACK.4. Depends on F-DATA-PACK.3 (per-PLM dataset cards,
lab PR #50). FARM-EXP.15 KNN-only baseline numbers are pending
(BLOCKED on FARM-EXP.13 compute); those cells carry TBD placeholders
and will be backfilled automatically once FARM-EXP.15 lands.

---

## 1. Data Sources

### UniProt and GOA releases

The `bench-v1` family uses GOA (Gene Ontology Annotation) lineage
deltas extracted from consecutive UniProt releases. Two snapshot
windows are relevant:

| Role | GOA window | Description |
|------|------------|-------------|
| Training window (13 pairs) | v160 through v226 | 13 consecutive delta pairs spanning ~10 years of annotation growth |
| Eval window (1 pair) | v226-v230 | Fixed held-out generalisation window |

A snapshot pair `vAAA-vBBB` contains GO annotations that appeared in
UniProt release `vBBB` and were absent in release `vAAA` (the lineage
delta). The 13 training pairs span v160 through v226 inclusive; the
eval pair is v226-v230.

The GOA snapshot identifiers used in this family are:
v160, v200, v210, v215, v220, v226, v230 (allowlisted) plus the
intermediate releases 165, 170, 175, 180, 185, 190, 195, 205, 211
(encoded as numeric strings in `manifest.json`, not as prose tokens).

### Ontology snapshot

All cells in the bench-v1 family share a single ontology snapshot:

| Field | Value |
|-------|-------|
| Ontology snapshot id | `35c3ad67-3002-47db-8f71-eeed69d22ad6` |
| Annotation source | `goa` |
| Eval window | v226-v230 |

---

## 2. Pipeline Version

The datasets are produced by PROTEA's `export_research_dataset`
operation. The producing version is pinned in each dataset's
`manifest.json` and per-dataset README.

| Field | Value |
|-------|-------|
| Producer | PROTEA `export_research_dataset` operation |
| Producer version | `0.8.0` |
| Producer git SHA (K=5 esm2_650m reference cell) | `b9c1dea8f20a1d5ec6e18689653f1ba9854dc2cf` |
| Schema SHA (canonical feature name digest) | `6d97a624b8a7` |
| Schema format | manifest schema version two (field: `schema_version`) |

The schema SHA `6d97a624b8a7` is a 12-hex prefix of the SHA-1 of the
canonical feature name list computed by PROTEA's
`parquet_export._compute_schema_sha`. All 8-PLM bench-v1 cells
produced by FARM-EXP.13 carry this same digest, confirming feature
schema stability across the full grid.

The producer git SHA is recorded per export job in `manifest.json`.
Consumers should validate both digests using the F-DATA-PACK.1
schema validator before training:

```bash
python scripts/validate_manifest.py --manifest path/to/manifest.json
```

---

## 3. Canonical 8-PLM Embedding Configurations

The bench-v1 family covers 8 protein language models. Each PLM maps to
a unique `embedding_config_id` in PROTEA's registry, pinned after the
audit described in ADR D35 (PROTEA PR #418). All 8 configs were
fully hydrated as of 2026-05-20 (no further `compute_embeddings` jobs
required).

| PLM key | HuggingFace / SDK checkpoint | embedding_config_id | Pool size |
|---------|------------------------------|---------------------|-----------|
| esm2_150m | facebook/esm2_t30_150M_UR50D | `500a0c59-be09-424d-9d51-b7997629c95a` | 527,855 |
| esm2_650m | facebook/esm2_t33_650M_UR50D | `c2e9dda3-e505-4170-b50d-435a451761ac` | 527,422 |
| esm2_3b | facebook/esm2_t36_3B_UR50D | `55e43f1c-1a3b-4b1d-88c0-26b433f5f673` | 551,918 |
| prot_t5 | Rostlab/prot_t5_xl_half_uniref50-enc | `084943c6-fec1-441d-bdc5-63b0268ada1b` | 527,855 |
| prostt5 | Rostlab/ProstT5 | `c0ae5b69-d6dc-41cf-a711-1739d3d2e170` | 527,424 |
| ankh_base | ElnaggarLab/ankh-base | `08234f06-ba76-4d7d-aaec-ae601096b4fa` | 527,424 |
| ankh_large | ElnaggarLab/ankh-large | `238f79b1-3068-4c6f-9013-5cc52b4f662b` | 527,424 |
| esmc_600m | esmc_600m (ESM SDK) | `2bf1e753-022f-44b8-a131-9a90acb4024e` | 527,424 |

Embeddings are stored as mean pools over the residue axis (CLS and EOS
stripped). The raw embedding dimension varies by PLM; all cells in the
bench-v1 family apply PCA(16) reduction before feature assembly (see
Section 5).

---

## 4. Split Methodology

The bench-v1 split uses temporal GO annotation lineage deltas as
train and eval partitions.

### Training pairs (13 consecutive delta windows)

The training window spans v160 through v226 in 5-release steps (one
irregular step at 205 to 211 to follow the UniProt release schedule).
The 13 pairs in order are:

160-165, 165-170, 170-175, 175-180, 180-185, 185-190, 190-195,
195-v200, v200-v210, v210-v215, v215-v220, v220-v226 and the
additional pair 205-211.

The exact list is recorded in the machine-readable form inside each
cell's `manifest.json` under `train_snapshot_pairs`. The above is a
human-readable summary; treat the manifest as the authoritative source.

### Eval pair (fixed)

v226-v230 — one held-out delta window shared across the full
bench-v1-K{3,5,10}-v226-lineage family. This ensures that every
cross-PLM and cross-K comparison sees the same ground truth.

### Why temporal splits?

Temporal splits avoid look-ahead leakage: any annotation in the eval
pair v226-v230 was absent from the annotation corpus at training
time. The lineage-delta formulation ensures that positive labels
correspond to genuinely new annotations, not re-annotations or
nomenclature changes.

---

## 5. PCA Fit Policy (Transductive)

Each bench-v1 cell applies PCA(16) to the PLM mean-pool embeddings.
The fit policy is **transductive**: PCA is fit on the full embedding
pool (train proteins union eval proteins) for each PLM, using the
embedding_config_id as the sole cache key.

| Property | Value |
|----------|-------|
| PCA components | 16 |
| Fit scope | Full pool (train + eval proteins, unsupervised) |
| Cache key | `embedding_config_id` |
| Label leakage | None (PCA is unsupervised; no annotation labels used) |
| Train-only refit | Optional future upgrade (deferred per 2026-05-20 decision) |

The transductive approach was accepted for the FARM-EXP.13 grid
because the PCA basis is unsupervised: it uses only the protein
embedding vectors, not the GO annotation labels. The full-pool PCA
captures the global embedding geometry, including proteins that only
appear in the eval split. A train-only refit would be a conservative
alternative but is not required for the publishable claim.

Reference: memory key `pca_transductive_decision_2026_05_20`.

---

## 6. Leakage-Free Note

### anc2vec construction artefact (PROTEA commit 223299c)

A pre-existing construction artefact in the anc2vec encoder caused
the feature `anc2vec_query_known_count` to act as a category-bucket
identifier when rows were replicated per GO category in the training
parquet. This is not temporal leakage (no future annotations are
revealed), but a replication artefact: the replication pattern for a
given (protein, aspect) pair was uniquely correlated with whether the
protein belonged to a genuine NK/LK category versus a synthetic
negative.

The fix (PROTEA commit `223299c`) filters (protein, aspect) rows by
genuine category membership before export, removing the bucket-effect.
All bench-v1 cells produced by FARM-EXP.13 are exported from a
PROTEA build that includes this fix.

**Story for thesis Chapter 6:** describe this as an "anc2vec
replication artefact", not as temporal leakage.

Reference: memory key `anc2vec_leakage_mechanism`.

---

## 7. KNN Baseline Numbers per Cell

The table below will be populated by FARM-EXP.15. All entries are
TBD placeholders pending FARM-EXP.13 compute completion. An automated
post-merge backfill will replace these once FARM-EXP.15 lands.

| PLM | K | Cell type | KNN-only Fmax (v226-v230) | Champion Fmax (v226-v230) | Delta | Sig95 |
|-----|---|-----------|---------------------------|---------------------------|-------|-------|
| esm2_650m | 5 | nk-mfo | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | nk-bpo | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | nk-cco | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | lk-mfo | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | lk-bpo | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | lk-cco | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | pk-mfo | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | pk-bpo | TBD | TBD | TBD | TBD |
| esm2_650m | 5 | pk-cco | TBD | TBD | TBD | TBD |
| esm2_150m | 5 | nk-mfo | TBD | TBD | TBD | TBD |
| esm2_150m | 5 | nk-bpo | TBD | TBD | TBD | TBD |
| esm2_150m | 5 | nk-cco | TBD | TBD | TBD | TBD |
| esm2_150m | 5 | lk-mfo | TBD | TBD | TBD | TBD |
| esm2_150m | 5 | lk-bpo | TBD | TBD | TBD | TBD |
| esm2_150m | 5 | lk-cco | TBD | TBD | TBD | TBD |
| esm2_3b | 5 | nk-mfo | TBD | TBD | TBD | TBD |
| esm2_3b | 5 | nk-bpo | TBD | TBD | TBD | TBD |
| esm2_3b | 5 | nk-cco | TBD | TBD | TBD | TBD |
| esm2_3b | 5 | lk-mfo | TBD | TBD | TBD | TBD |
| esm2_3b | 5 | lk-bpo | TBD | TBD | TBD | TBD |
| esm2_3b | 5 | lk-cco | TBD | TBD | TBD | TBD |
| prostt5 | 5 | nk-mfo | TBD | TBD | TBD | TBD |
| prostt5 | 5 | nk-bpo | TBD | TBD | TBD | TBD |
| prostt5 | 5 | nk-cco | TBD | TBD | TBD | TBD |
| prostt5 | 5 | lk-mfo | TBD | TBD | TBD | TBD |
| prostt5 | 5 | lk-bpo | TBD | TBD | TBD | TBD |
| prostt5 | 5 | lk-cco | TBD | TBD | TBD | TBD |
| ankh_base | 10 | nk-mfo | TBD | TBD | TBD | TBD |
| ankh_base | 10 | nk-bpo | TBD | TBD | TBD | TBD |
| ankh_base | 10 | nk-cco | TBD | TBD | TBD | TBD |
| ankh_large | 5 | nk-mfo | TBD | TBD | TBD | TBD |
| ankh_large | 10 | nk-mfo | TBD | TBD | TBD | TBD |
| prot_t5 | pending EXP.13 | all | TBD | TBD | TBD | TBD |
| esmc_600m | pending EXP.13 | all | TBD | TBD | TBD | TBD |

**Known reference (prostt5, K=5, v226-lineage, multi-seed pilot):**

The binary-objective multiseed pilot on `bench-v1-K5-v226-lineage-prostt5`
(recipe: `run--plm=prostt5--k=5--rr=lgbm.binary--feat=lean+lin+emb--eval=bench-v1-K5-v226-lineage-prostt5--prop=fill--ens=none`,
3 seeds: 42, 137, 244) produced:

| Cell | Fmax (mean, 3 seeds) | 95% CI half |
|------|----------------------|-------------|
| nk-mfo | 0.7408 | 0.0030 |
| nk-bpo | 0.5887 | 0.0035 |
| nk-cco | 0.7980 | 0.0025 |
| lk-mfo | 0.6821 | 0.0012 |
| lk-bpo | 0.6678 | 0.0048 |
| lk-cco | 0.7973 | 0.0027 |

NK+LK unweighted mean: **0.7291 +/- 0.0028** (publishable Chapter 6 claim).
Reference: memory key `v27_binary_multiseed_2026_05_18`, lab PR #29.

---

## 8. FAIR Checklist

### Findable

| Criterion | Status | Notes |
|-----------|--------|-------|
| F1: Dataset has a persistent identifier | Pending | DOI: TBD (Zenodo upload in F-DATA-PACK.5) |
| F2: Data described with rich metadata | Done | Per-dataset README + dataset card per PLM |
| F3: Metadata clearly includes identifier | Pending | Will reference Zenodo DOI once assigned |
| F4: Dataset registered in searchable resource | Pending | Zenodo deposit in F-DATA-PACK.5 |

### Accessible

| Criterion | Status | Notes |
|-----------|--------|-------|
| A1: Retrievable by identifier using open protocol | Pending | Zenodo URL TBD, assigned in F-DATA-PACK.5 |
| A1.1: Protocol is open, free, and universally implementable | Planned | Zenodo HTTPS |
| A1.2: Protocol allows authentication where necessary | N/A | Public deposit planned |
| A2: Metadata accessible even if data unavailable | Planned | Zenodo metadata persists independently of parquet files |

Storage backend during active research: MinIO (`s3://protea/datasets/`).
Zenodo archival is the persistent public endpoint and is handled by
F-DATA-PACK.5.

### Interoperable

| Criterion | Status | Notes |
|-----------|--------|-------|
| I1: Data uses a formal, accessible, shared representation | Done | Apache Parquet (columnar, open standard) |
| I2: Data uses FAIR vocabularies | Done | GO term IDs (OBO ontology), UniProt accessions (canonical IDs) |
| I3: Dataset includes qualified references to other datasets | Done | manifest.json references embedding_config_id + ontology_snapshot_id |

File formats used:

- Train and eval splits: Apache Parquet (snappy compression)
- Manifest: JSON (schema format version two, validated by F-DATA-PACK.1)
- Dataset card: Markdown with YAML front matter (HuggingFace card
  format, per-PLM)
- Per-dataset README: Markdown (auto-generated by
  `scripts/generate_dataset_readme.py`)

### Reusable

| Criterion | Status | Notes |
|-----------|--------|-------|
| R1: Data richly described with plurality of accurate, relevant attributes | Done | schema_sha + manifest_sha + producer git SHA + export job id pinned |
| R1.1: Data released with clear and accessible data usage license | Done | CC BY 4.0 on all bench-v1 cells |
| R1.2: Data associated with detailed provenance | Done | This document plus per-dataset READMEs and dataset cards |
| R1.3: Data meets domain-relevant community standards | Partial | Uses standard GO/UniProt identifiers; CAFA-compatible eval; Zenodo deposit pending |

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

Author and creator: Francisco Miguel Perez Canales, doctoral thesis,
PROTEA (Protein functional Embedding-based Annotation), Universidad de
Sevilla.

Citation (before Zenodo DOI is available): cite PROTEA repository at
producer git SHA `b9c1dea8f20a1d5ec6e18689653f1ba9854dc2cf` (reference
cell esm2_650m K=5) or the relevant cell-specific SHA from each
dataset's manifest.

---

## 9. Cross-References

### Thesis

The dataset family is the primary research deliverable for the doctoral
thesis. The publishable results grid (full PLM x K x recipe, no
champion selection) is reported in Chapter 6 and Appendix A.

- Chapter 6: cafaeval Fmax grid on v226 through v230, full
  (PLM x K x recipe) report per memory key
  `multi_plm_report_full_grid`.
- Appendix A: full provenance manifest table (mirrors Section 2-5 of
  this document).

The thesis Chapter 6 was refreshed by thesis PR #51. Any update to
the KNN baseline TBD cells in Section 7 should be accompanied by a
corresponding update to the Chapter 6 results table in the thesis
manuscript.

### Lab artefacts

| Artefact | Location |
|----------|----------|
| Per-PLM dataset cards | `dataset_cards/<plm>_card.md` |
| Per-dataset READMEs | `datasets/<name>/README.md` |
| Manifest schema validator | `scripts/validate_manifest.py` |
| README generator | `scripts/generate_dataset_readme.py` |
| FAIR packaging slices | F-DATA-PACK.1 (validator), F-DATA-PACK.2 (README gen), F-DATA-PACK.3 (dataset cards), F-DATA-PACK.4 (this document), F-DATA-PACK.5 (Zenodo deposit, pending) |

### PROTEA ADRs

- ADR D35: canonical 8-PLM embedding config audit (PROTEA PR #418)
- PROTEA commit `223299c`: anc2vec replication artefact fix
