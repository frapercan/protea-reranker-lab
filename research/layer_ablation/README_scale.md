# At-scale representation ablation: launch plan

The crown (`crown_train.py` -> `crown_result.json`) trained every arm on the 15k
reference and its L48 control hit mean9 f_micro_w **0.1425** vs the served champion's
**0.2150**: starved 6.7x. The apples control (`crown_control_apples.py` ->
`crown_control_apples.json`) proved the gap was data starvation: the FULL learned
encoder trained on the champion's 100k pool, scored on the SAME 15k reference,
reached **0.2197**. This run reruns the crown's ARM comparison at that scale.

## What it answers

At 100k training (no starvation):
1. Does **L10-std** beat **L48** (champion base) AND the champion's 0.2150?
2. Does a **learned softmax mix** over layers {10,19,48} separate from L10-std?

Retests the crown's *starved* verdicts: z-score was the lever (L10 raw did not beat
L48, L10-std did, p=6.9e-28); the mix did NOT rediscover L10 (near-uniform, mild L19
peak). Holds at scale, or not.

## Design (the one structural change from the crown)

The head trains on the **100k pool** base; it is then applied to the crown's own
**7,401 queries** and **15,000 reference** proteins for scoring. Training pool, query
set, and kNN reference are three DISTINCT sets (in the crown the training base and the
kNN reference were the same 15k). z-score stats are fit on the pool (the training base).

## Prerequisites (all present as of 2026-07-11 ~04:00 except the pool embeddings)

- `scale_pool_meta.json`, `scale_pool_seqs.json` (100k accs + seqs)  ok
- `scale_pool_go.json` (100k -> raw GO leaves, read-only DB pull via
  `scale_pool_go_build.py`; 1,020,937 leaves, mean 10.2/protein)  ok
- `emb_ankh_base/layer_{10,19,48}.npy` + `meta.json` (7,401 queries)  ok
- `ref_emb/layer_{10,19,48}.npy` + `meta.json` (15,000 reference)  ok
- `ref_go.json`, `seqs.json`, `knn_confirm_results.json`  ok
- **`scale_pool_emb/layer_{10,19,48}.npy` + `meta.json`** -- being extracted by
  `scale_extract.py` (log: `scale_extract.log`), written only at the END. This is the
  one gate.

## Launch (when `scale_pool_emb/layer_48.npy` exists)

The GPU must be free (the extraction must have finished; a busy GPU also fails CUDA
tests). Run on the lab venv, background with a durable log:

```
cd /home/frapercan/Thesis2/storage/layer_ablation
PYTHONPATH=/home/frapercan/Thesis2/repositories/protea-reranker-lab/src \
  /home/frapercan/Thesis2/repositories/protea-reranker-lab/.venv/bin/python \
  scale_train.py 2>&1 | tee scale_train.log
```

Arms/seeds via env (`SCALE_ARMS`, `SCALE_SEEDS`, `SCALE_EPOCHS`; `SCALE_SKIP_CAFA=1`
for a fast smoke). Default: `L48,L10,L10-std,mix-learned` x seeds 42,43,44.

Smoke first (one arm, 2 epochs, skip cafaeval) to catch any 100k GPU-memory issue
before the full ~1h run:
```
SCALE_ARMS=L48 SCALE_SEEDS=42 SCALE_EPOCHS=2 SCALE_SKIP_CAFA=1 <same invocation>
```

## Outputs (receipt)

- `scale_result.json` -- 9-cell f_micro_w per arm (mean/std over seeds), control check
  vs champion, learned softmax weights, per-protein IA-Fmax + bootstrap CIs,
  Wilcoxon+Holm, length/identity stratification.
- `scale_perprot.npz` -- per-(cell,protein) IA-Fmax vectors per arm.

## Read the result against these anchors (verify, do not assume)

- `crown_result.json` (starved 15k): L48 0.1425, the z-score/mix verdicts.
- `crown_control_apples.json`: full encoder at 100k = 0.2197.
- `knn_confirm_results.json["d8979601-learned"]`: the served champion, mean9 0.2150.
- Sealed thesis number 0.4063 is the RERANKED pipeline, a different measurement; these
  KNN-only mean9 numbers are the retrieval-encoder lever, not the headline.
