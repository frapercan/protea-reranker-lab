# Build, push, and submit the fullgo ensemble LAFA container

Status: weights persisted (`storage/fullgo_models/`), inference + Docker scaffold here.
REMAINING (heavy, needs GPU build + ghcr + the maintainer): finish the two
`NotImplementedError` hooks, build, push, and submit. This file is the runbook.

## Realities (read first)
- The `Sep_2025_Mar_2026` window is PUBLISHED/CLOSED. A new container scores on the
  NEXT open window (different targets). Our offline 0.349 on Sep_2025_Mar_2026 is the
  retrospective benchmark; a submission gives a live position on the next window.
- Heavy image (~20 GB of PLM weights). The maintainers run on A100; do NOT expect it
  to fit our 12 GB box for the embedding step at scale.

## 1. Finish predict.py
- `embed_queries`: wire to protea-backends embedders (mean pooling, the 6 configs in
  `fullgo/config.yaml`, hstack in that order -> 8320-d).
- `knn_features`: numpy cosine KNN (K=30) over the bundled reference Ankh-base block;
  transfer ref labels; composite = weighted mean of [embedding_similarity, vote];
  vote = fraction of K neighbours carrying the term.

## 2. Deployable feature set (train/serve parity)
The lab 0.349 GBMs use [knn, dist, id_nw, id_sw, tax, vote, clf, knn_p, clf_p, IA, lfreq].
The container cannot cheaply produce alignment (NW/SW) or taxonomy features. RETRAIN the
3 GBMs on the 7-feature deployable set [knn, vote, clf, knn_p, clf_p, IA, lfreq] and
re-seal on 7401 to confirm the score (expected ~0.34-0.35; those features were
low-importance). Replace `ensemble_gbm_*.txt` with the retrained boosters.

## 3. Per-category routing
LAFA scores NK/LK/PK separately but the container does not know a query's category at
run time. Options: (a) emit one prediction set and let the harness route (use the NK
booster, the most general), or (b) emit all three and let the maintainer pick per frame.
Decide and document; current predict.py uses the NK booster as default.

## 4. Reference bundle (bind-mounted at /app/data)
`reference_pool.npz` (t0 Ankh-base embeddings + accessions + propagated labels for KNN),
`IA.tsv`. Build it from the v227 frozen store (see fullgo/extract_*.py).

## 5. Build / push / submit
    cp ../../../storage/fullgo_models/* models/        # bake weights
    docker build -t ghcr.io/frapercan/protea/fullgo-ensemble-v1 .
    docker push ghcr.io/frapercan/protea/fullgo-ensemble-v1
    # submit: PR/issue to github.com/anphan0828/CAFA_forever adding the method
    # (image + resource + flags), per LAFA_SUBMISSION_GUIDE.md.

## 6. Local smoke test
    docker run --gpus all -v $PWD/testdata:/app/data:ro -v $PWD/out:/app/output:rw \
      ghcr.io/frapercan/protea/fullgo-ensemble-v1 \
      -q /app/data/queries.fasta -o /app/output/predictions.tsv
