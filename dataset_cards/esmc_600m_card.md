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
  - esmc
  - esm-cambrian
  - reranker
  - knn
  - lightgbm
  - benchmark
pretty_name: "bench-v1 v226-lineage esmc_600m (K=3,5,10)"
zenodo_doi: TBD
---

# bench-v1-K{3,5,10}-v226-lineage-esmc_600m

Dataset family produced by PROTEA's `export_research_dataset` operation
for the ESM-C 600M (Cambrian) protein language model. Each cell in the K
axis is a frozen LightGBM feature dataset covering one (K, split-window)
combination. These are the primary research deliverables for the thesis
per the 2026-05-25 decision (memory `dl-postponed-2026-05-25`).

## PLM identity

| Field | Value |
| --- | --- |
| PLM key | `esmc_600m` |
| SDK checkpoint | `esmc_600m` (EvolutionaryScale ESM SDK) |
| Architecture | ESM-C 600M (Cambrian; loaded via ESM SDK, not HuggingFace) |
| Embedding dimension (raw) | 1152 |
| Pooling strategy | Mean pool over residue axis (CLS and EOS stripped) |
| PCA applied | PCA(16) per config, fit on full pool (transductive, unsupervised) |
| Embedding config id | `2bf1e753-022f-44b8-a131-9a90acb4024e` |

## K-axis variants

All three cells share the same embedding config, ontology snapshot
(`35c3ad67-3002-47db-8f71-eeed69d22ad6`), annotation source (`goa`),
and eval window (`v226-v230`). All K cells are pending EXP.13
completion.

| Variant | Train rows | Eval rows | Schema sha | Status |
| --- | --- | --- | --- | --- |
| bench-v1-K3-v226-lineage-esmc_600m | TBD | TBD | 6d97a624b8a7 | Pending EXP.13 completion |
| bench-v1-K5-v226-lineage-esmc_600m | TBD | TBD | 6d97a624b8a7 | Pending EXP.13 completion |
| bench-v1-K10-v226-lineage-esmc_600m | TBD | TBD | 6d97a624b8a7 | Pending EXP.13 completion |

Row counts will be updated once EXP.13 finishes. Based on comparable
PLMs (527 424 proteins in the pool), expected K5 train rows are in the
range 24-26M and eval rows around 1.0-1.2M.

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
`embedding_config_id = 2bf1e753-022f-44b8-a131-9a90acb4024e`.

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

**All K cells pending.** EXP.13 export jobs for this PLM are in-flight
as of 2026-05-25. Row counts are placeholders.

**ESM SDK dependency.** Embeddings were produced using the
EvolutionaryScale proprietary ESM SDK (`esm.models.esmc.ESMC`), not the
HuggingFace transformers library. The SDK is not open-source; users who
want to reproduce the embedding step must obtain SDK access separately.

## License and citation

CC BY 4.0. Author: Francisco Miguel Perez Canales, doctoral thesis,
PROTEA (Protein functional Embedding-based Annotation). Zenodo DOI
placeholder; see field `zenodo_doi: TBD` above.
