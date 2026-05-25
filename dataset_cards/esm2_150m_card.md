---
annotations_creators:
  - machine-generated
language_creators:
  - other
language: []
license: cc-by-4.0
multilinguality: monolingual
size_categories:
  - 10M<n<100M
source_datasets: []
task_categories:
  - text-classification
task_ids:
  - multi-class-classification
tags:
  - protein-function
  - go-annotation
  - protein-language-model
  - esm2
  - reranker
  - knn
  - lightgbm
  - benchmark
pretty_name: "bench-v1 v226-lineage esm2_150m (K=3,5,10)"
zenodo_doi: TBD
---

# bench-v1-K{3,5,10}-v226-lineage-esm2_150m

Dataset family produced by PROTEA's `export_research_dataset` operation
for the ESM-2 150M protein language model. Each cell in the K axis is a
frozen LightGBM feature dataset covering one (K, split-window)
combination. These are the primary research deliverables for the thesis
per the 2026-05-25 decision (memory `dl-postponed-2026-05-25`). The
neural-head deep-learning line was postponed because it offers no clear
win over the LightGBM champion on the v226 to v230 generalisation window;
the curated multi-PLM x K dataset family PROTEA produces was elevated to
the primary contribution.

## PLM identity

| Field | Value |
| --- | --- |
| PLM key | `esm2_150m` |
| HuggingFace checkpoint | `facebook/esm2_t30_150M_UR50D` |
| Architecture | ESM-2 (transformer encoder, 30 layers) |
| Embedding dimension (raw) | 640 |
| Pooling strategy | Mean pool over residue axis (CLS and EOS stripped) |
| PCA applied | PCA(16) per config, fit on full pool (transductive, unsupervised) |
| Embedding config id | `500a0c59-be09-424d-9d51-b7997629c95a` |

## K-axis variants

All three cells share the same embedding config, ontology snapshot
(`35c3ad67-3002-47db-8f71-eeed69d22ad6`), annotation source (`goa`),
and eval window (`v226-v230`). Row counts differ because K controls
how many KNN neighbours are retrieved per query protein and GO term.

| Variant | Train rows | Eval rows | Schema sha | Status |
| --- | --- | --- | --- | --- |
| bench-v1-K3-v226-lineage-esm2_150m | TBD | TBD | 6d97a624b8a7 | Pending EXP.13 completion |
| bench-v1-K5-v226-lineage-esm2_150m | 25,292,914 | 1,113,367 | 6d97a624b8a7 | Available |
| bench-v1-K10-v226-lineage-esm2_150m | 36,281,441 | 1,625,022 | 6d97a624b8a7 | Available |

Note: K3 row counts will be updated once EXP.13 finishes. All three
cells carry the same canonical schema sha once produced.

## Split methodology

The dataset uses temporal GO annotation lineage deltas as train and eval
splits. A snapshot pair `vNNN-vMMM` contains GO annotations that appeared
in UniProt release `vMMM` and were absent in release `vNNN`.

| Split | Description |
| --- | --- |
| Train | 13 consecutive GOA release delta pairs spanning v160 through v226 |
| Eval | v226-v230 (1 pair, fixed across the bench-v1 K-variant family) |

## Embedding pool size

Approximately 527,855 proteins with embeddings stored under
`embedding_config_id = 500a0c59-be09-424d-9d51-b7997629c95a`. The pool
was fully hydrated by 2026-05-20 (ADR D35, PROTEA PR #418).

## Schema

Each row is one (query protein, retrieved GO term) candidate produced by
K nearest neighbour retrieval against the PLM embedding pool. The binary
label records whether the annotation appeared in the eval lineage delta.

Top-level column families (canonical list pinned by schema sha
`6d97a624b8a7`):

- identifiers: `protein_id`, `go_term`, `aspect`, `eval_snapshot_pair`
- retrieval: KNN distances, retrieval ranks, lineage scores
- alignment: pairwise NW/SW alignment statistics
- taxonomy: lineage agreement features for query and retrieved proteins
- embeddings: anc2vec GO term features (200 dim) plus PCA(16) of the
  PLM mean pool
- labels: binary annotation outcome

## Caveats

**PCA transductive fit.** The PCA(16) basis is fit on the union of train
and eval embedding pools (unsupervised; no label leakage). A train-only
refit is deferred per memory `pca-transductive-decision-2026-05-20`.

**anc2vec construction artefact.** The anc2vec encoder historically
replicated rows per GO category, making `anc2vec_query_known_count`
behave as a category-bucket identifier under certain training splits.
PROTEA commit `223299c` filters (protein, aspect) rows by genuine
category membership. Consumers training on this dataset inherit that fix.
See memory `anc2vec-leakage-mechanism` for the full mechanism.

**K3 cells pending.** Row counts for K=3 are placeholders. Final numbers
arrive when EXP.13 finishes (ETA 2026-05-25).

## Producer provenance (K5 cell, reference)

| Field | Value |
| --- | --- |
| Producer version | 0.8.0 |
| Producer git sha | `e130642deae502e20f0861f37cc22254e7e7cbe0` |
| Export job id | `eeb1a7a8-d3a8-4de6-9ea2-6dd95b5f1d4f` |

## License and citation

CC BY 4.0. Author: Francisco Miguel Perez Canales, doctoral thesis,
PROTEA (Protein functional Embedding-based Annotation). Zenodo DOI
placeholder; see field `zenodo_doi: TBD` above. Until the DOI lands,
cite the PROTEA repository at producer git sha
`e130642deae502e20f0861f37cc22254e7e7cbe0`.
