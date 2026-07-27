# Capturing BP with the resources we have: technique levers only

Constraint set by the author (2026-07-16): go after the two missing cells (LK-BPO,
PK-BPO) using **techniques and simple data only**. No structures, no text, no new
modality. So every lever below reuses the features we already export.

Baseline to beat, all measured on PK-BPO: reranker **AUC 0.7903 -> f_micro_w 0.1255**;
the pool's oracle ceiling is **0.6077** (we capture 20.7%); **+0.044 AUC is worth
+0.112 f_micro_w**. Gap to TransFew: **+0.076**.

## The two suspicions worth probing first

### 1. The model pools aspects (per-category, not per-cell)

`train_rerank_227230.py` trains one booster per **category** (nk/lk/pk) with MFO, BPO
and CCO candidates **pooled** into the same model and the same trees. BP does not
share signal structure with MF: the LOFO grid already showed the regimes differ
(classifier carries NK/LK, association and anc2vec_query carry PK). A booster forced
to serve three aspects at once spends capacity reconciling them.

A dedicated **PK-BPO** model is a pure technique change: same rows, same features,
narrower target.

### 2. The objective does not match the metric (the stronger suspicion)

This is the one that best explains why a global **AUC of 0.79** converts into only
**f_micro_w 0.126**:

- **LambdaMART with `lambdarank` + `ndcg_eval_at [5,10]`, grouped by
  (snapshot_pair, protein, aspect)** optimises the order of candidates **within one
  protein**, and only the head of that list.
- **`f_micro_w` sweeps ONE GLOBAL threshold across every protein at once.** What it
  needs is that a score of 0.7 means the same thing for protein A and protein B.

LambdaMART gives no such guarantee: it is invariant to any per-group monotone
rescaling. Ranking each protein perfectly while leaving the scores mutually
incomparable produces exactly our symptom - respectable AUC, poor thresholded F.

The technique answer is a **globally calibrated scorer**: `binary` logloss (or
lambdarank followed by an explicit cross-protein calibration), which is trained to
output a comparable probability rather than a within-list order.

### 3. The learned representation never reaches the ranker (author's question, verified)

The champion is a **learned k-WTA sparse encoder over Ankh-base** (`d8979601`). In this
run it is used **only to retrieve neighbours**. The reranker never sees it:

- the 16 `emb_pca_query_*` columns, the only features carrying the learned code into
  the model, are **all NaN** in this export - dead;
- what survives are **collapsed scalars**: `distance`, `neighbor_*`, `k_position`,
  `vote_count`. A whole representation compressed to a handful of numbers.

We spend the thesis learning a representation and then discard it exactly at the step
where the measurements say we lose (ranking, 20.7% capture).

Two honest qualifications before treating this as free money:

- `emb_pca_query_*` is the **query's** embedding, which is **constant across that
  protein's candidates**. Even alive it cannot discriminate *between* candidates, and
  `lambdarank` is invariant to it by construction. Its value would be for
  **cross-protein comparability** - which is precisely what lever 2 says is broken.
- The discriminative form does not exist as a feature at all: a **query x candidate
  interaction in the learned space** (e.g. the query code against a term-level
  aggregate of the neighbours that voted the term, or against a learned term
  embedding). That is what would separate candidates of the same protein.

Fits the constraint exactly: the 527k vectors are already materialised. This is
technique, not new data.

## Other technique levers inside the constraint (untested, ranked)

3. **Within-protein rank features.** Re-express existing features as their rank or
   z-score *inside each protein's candidate list* (distance rank, vote-fraction rank,
   classifier-score rank). This hands the model the relative structure it currently
   has to infer, and costs no new data.
4. **Drop classifier-proposed candidates from the pool.** Measured: unioning them
   takes the pool from 62 to 114 candidates/query and dilutes mean9 by ~0.058
   (KNN-only scores 0.4594 vs full-pool 0.401). Classifier stays as a *feature*, not
   as a candidate generator. Config-level, already quantified.
5. **Class weighting / positive-rate handling.** PK-BPO positives are 1.06% in train,
   2.47% in the eval pool. `scale_pos_weight` or focal-style weighting is untried.
6. **Per-protein score normalisation at inference** (a cheaper cousin of lever 2):
   keep lambdarank, then map each protein's scores through a calibrator so the global
   threshold is meaningful.

## Ruled out by measurement (do not revisit without new evidence)

- co-occurrence candidate expansion: +0.0021 (recall 0.32 -> 0.48)
- protst as a feature: +0.0016 on PK-BPO
- InterPro graft: **negative** on BP (0.140 vs 0.218)
- GO-DAG hierarchical proximity: AUC 0.5501, blending adds +0.0002
- a better/bigger Ankh: a representation gain re-ranks the same candidates, and recall
  is not the binding constraint

Receipts: `BP_WALL_CHARACTERIZATION.md`, `storage/cooc_experiment/`.
