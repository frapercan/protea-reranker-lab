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
  - ankh
  - reranker
  - knn
  - lightgbm
  - benchmark
pretty_name: "bench-v1 v226-lineage ankh_large (K=3,5,10)"
zenodo_doi: TBD
---

# bench-v1-K{3,5,10}-v226-lineage-ankh_large

Dataset family produced by PROTEA's `export_research_dataset` operation
for the Ankh-large protein language model. Each cell in the K axis is a
frozen LightGBM feature dataset covering one (K, split-window)
combination. These are the primary research deliverables for the thesis
per the 2026-05-25 decision (memory `dl-postponed-2026-05-25`).

## PLM identity

| Field | Value |
| --- | --- |
| PLM key | `ankh_large` |
| HuggingFace checkpoint | `ElnaggarLab/ankh-large` |
| Architecture | Ankh-large (T5 backbone, larger capacity than ankh-base) |
| Embedding dimension (raw) | 1536 |
| Pooling strategy | Mean pool over residue axis (last padding token stripped) |
| PCA applied | PCA(16) per config, fit on full pool (transductive, unsupervised) |
| Embedding config id | `238f79b1-3068-4c6f-9013-5cc52b4f662b` |

## K-axis variants

All three cells share the same embedding config, ontology snapshot
(`35c3ad67-3002-47db-8f71-eeed69d22ad6`), annotation source (`goa`),
and eval window (`v226-v230`).

| Variant | Train rows | Eval rows | Schema sha | Status |
| --- | --- | --- | --- | --- |
| bench-v1-K3-v226-lineage-ankh_large | TBD | TBD | 6d97a624b8a7 | Pending EXP.13 completion |
| bench-v1-K5-v226-lineage-ankh_large | 23,791,790 | 1,048,067 | 6d97a624b8a7 | Available |
| bench-v1-K10-v226-lineage-ankh_large | 32,251,063 | 1,447,450 | 6d97a624b8a7 | Available |

Note: K3 row counts will be updated once EXP.13 finishes.

## Split methodology

The dataset uses temporal GO annotation lineage deltas as train and eval
splits. A snapshot pair `vNNN-vMMM` contains GO annotations that appeared
in UniProt release `vMMM` and were absent in release `vNNN`.

| Split | Description |
| --- | --- |
| Train | 13 consecutive GOA release delta pairs spanning v160 through v226 |
| Eval | v226-v230 (1 pair, fixed across the bench-v1 K-variant family) |

## Embedding pool size

Approximately 527,424 proteins with embeddings stored under
`embedding_config_id = 238f79b1-3068-4c6f-9013-5cc52b4f662b`.

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

**anc2vec construction artefact.** PROTEA commit `223299c` filters
(protein, aspect) rows by genuine category membership. See memory
`anc2vec-leakage-mechanism`.

**K3 cells pending.** Row counts for K=3 are placeholders. Final
numbers arrive when EXP.13 finishes (ETA 2026-05-25).

## License and citation

CC BY 4.0. Author: Francisco Miguel Perez Canales, doctoral thesis,
PROTEA (Protein functional Embedding-based Annotation). Zenodo DOI
placeholder; see field `zenodo_doi: TBD` above.
