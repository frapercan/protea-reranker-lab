# F-LAFA-IA.1b: v227 probe re-export + IA-weighted eval

This directory holds the eval harness for the v227 probe cell
(`bench-v1-K3-v227-lineage-prostt5`, train cutoff v227, eval band
v227->v230) and the operational notes from the probe run.

## Eval harness

`eval_v227_probe.py` evaluates the re-exported v227 dataset against the
LAFA protocol pinned in `../lafa_ia_v227_protocol/` (prop=fill,
norm=cafa, no_orphans, `ia=datasets/ia/IA-swissprot-exp-v227.txt`).

Per (cell, arm) it records four numbers: internal Fmax (lab grid),
cafaeval Fmax (`f_micro`), wFmax (IA-weighted micro Fmax `f_micro_w`,
the LAFA headline), and S_min (`s`). Two arms: the binary-objective
reranker (seed 42) and the KNN baseline (raw `neighbor_vote_fraction` score;
cafaeval Fmax is rank-invariant to a monotone score scale).

Run (after the dataset is downloaded into `datasets/<name>/`):

```
.venv/bin/python experiments/lafa_ia_v227_probe/eval_v227_probe.py
```

The cafaeval path was smoke-validated end-to-end against the
v226 K3 prostt5 KNN baseline (nk-mfo): f_micro 0.4675, f_micro_w 0.4273,
s 4.4527 with the `ia=` table.

## Re-export dispatch (v227 payload)

The export payload mirrors the EXP.13 v226 grid (compute_alignments,
compute_taxonomy, expand_votes_to_ancestors, use_embedding_pca all
true, faiss backend) and only moves the band: `train_versions` appends
227 (last train pair `v226-v227`), `test_versions=[230]`, giving the
eval pair `v227->v230`. The `220-226`, `226-227` and `227-230`
EvaluationSets are all materialised; v227 (`c905dffa`) shares the
v226/v230 ontology snapshot (`35c3ad67`).

## Operational note: postgres /dev/shm

The export's first query (an embedding-count aggregate) spills a
parallel-worker shared-memory segment. On the native Docker engine the
postgres container ships with the default `/dev/shm` of 64 MB, which is
too small and raises `psycopg.errors.DiskFull` ("could not resize shared
memory segment ... No space left on device", CONTEXT: parallel worker).
This is NOT root-disk exhaustion (root had 413 GB free).

Fix: recreate the postgres container with `--shm-size=1g` (or set
`shm_size: 1g` in the compose service). A reload-free stopgap is
`ALTER DATABASE protea SET max_parallel_workers_per_gather = 0`, which
removes the parallel-worker segment entirely.

This blocks the F-LAFA-IA.1c 24-cell fanout: confirm the postgres
container `/dev/shm` is >= 1 GB before fanning out.
