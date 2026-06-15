# Reproduce the 0.381 champion ensemble (ties TransFew #1)

All paths are the dev-box absolute paths used during the 2026-06-14/15 build; adjust as needed. Requires
the PROTEA Postgres up (embeddings + annotations), the deploy venv (torch+cuda, lightgbm, cafaeval,
psycopg, scipy), and the LAFA harness artifacts in `config.yaml`.

PY=/home/frapercan/Thesis2/worktrees/protea-deploy/.venv/bin/python

## Frames

- **TEST (LAFA):** t0 = v227 (2025-09-04), t1 = v230 (2026-03-04). Eval = the 7401 official targets.
- **SELECT (validation):** t0 = v220, t1 = v227. Eval = the 220->227 delta proteins (eval set a3be0a6d).

## 1. KNN composite predictions (from the PROTEA platform)

The KNN side is produced by the platform (predict_go_terms, Ankh-base K30, composite scoring config
bae5ece3) and exported as `score.tsv`. TEST: prediction set 12739db3 (query set acc27f47 = 7401).
SELECT: prediction set 746e68e4 (ref v220). Export via `GET /scoring/prediction-sets/{id}/score.tsv`.
-> select_knn_composite.tsv (SELECT), canon_composite.tsv (TEST).

## 2. Classifier data (6-PLM embeddings + propagated t0 labels)

    # TEST frame (v227 labels, 7401 eval): base PLM then append the other 5
    $PY extract_base_plm.py        # -> m0_data.npz  (Ankh-base + v227 labels + 7401 eval)
    $PY extract_extra_plms.py      # -> m0v3_data.npz (append ESM2-3B/Ankh-large/ESM2-650M/ESMC/ProtT5)
    # SELECT frame (v220 labels, 220->227 eval proteins)
    $PY extract_select_frame.py    # -> sel_data.npz

## 3. Train the M2 classifier per seed and seed-average (5 seeds)

The champion classifier is the M2 anc2vec hybrid, seed-averaged over 5 seeds (base + 7 + 137 + 23 + 91).
Run each seed on both frames, then combine the consensus union (score = sum/n):

    # TEST frame
    $PY train_classifier_m2.py m0v3_data.npz m2_pred_base.tsv
    for S in 7 137 23 91; do $PY train_classifier_m2.py m0v3_data.npz m2_pred_s$S.tsv --seed $S; done
    $PY seed_average.py m2_seedavg5_pred.tsv m2_pred_base.tsv m2_pred_s7.tsv m2_pred_s137.tsv m2_pred_s23.tsv m2_pred_s91.tsv
    # SELECT frame
    $PY train_classifier_m2.py sel_data.npz sel_m2_base.tsv
    for S in 7 137 23 91; do $PY train_classifier_m2.py sel_data.npz sel_m2_s$S.tsv --seed $S; done
    $PY seed_average.py sel_m2_seedavg5.tsv sel_m2_base.tsv sel_m2_s7.tsv sel_m2_s137.tsv sel_m2_s23.tsv sel_m2_s91.tsv

## 4. Self-prior stream (GOA non-experimental t0, propagated)

`select_selfprior_leaf.tsv` (SELECT, v220) and `goa_nonexp_7401.tsv` (TEST) are the propagated
non-experimental t0 GOA annotations per protein, dumped from Postgres. Used as union candidates + the two
GBM columns `sp`, `sp_p`.

## 5. Cross-aspect association prior (targets LK/PK)

Per-protein t0 KNOWN experimental terms come from `eval_labels.py` (one npz per frame:
`eval_labels_sel.npz`, `eval_labels_7401.npz`; these partition exactly on NK 0% / LK 100% / PK 100%, the
leakage check). `assoc_feature.py` builds the training co-occurrence and emits, per candidate term, the
total and cross-aspect association P(t|known k):

    $PY assoc_feature.py sel_data.npz  eval_labels_sel.npz  assoc_sel.tsv
    $PY assoc_feature.py m0v3_data.npz eval_labels_7401.npz assoc_7401.tsv

## 6. Champion ensemble (fit SELECT, seal TEST) -> 0.390

    $PY ensemble_seal.py     # NK 0.472 / LK 0.481 / PK 0.217 / MEAN 0.390 (OUTRIGHT #1)
                             # writes ensemble_gbm_{NK,LK,PK}.txt + feature_spec.json to storage/fullgo_models/

## 7. Standalone scoring with the exact harness

    $PY evaluate_exact_harness.py --pred m2_seedavg5_pred.tsv --toi   # 5-seed classifier alone on 7401

## Notes

- `v227_exp_freq.tsv` / `v220_exp_aspect.tsv` (term frequency for the log_freq feature) are dumped from
  Postgres (see the queries inline in the scripts). `select_pk_known.tsv`, `select_gt_{cat}.tsv` come
  from the platform eval-set download endpoints (eval set a3be0a6d).
- The association feature is leakage-clean by construction: it uses only each protein's t0-known
  experimental terms (the CAFA "known" partition), so its GBM importance is exactly 0 on NK.
- LightGBM run-to-run variance is small (the 0.390 mean reproduces at 0.3902-0.3907).
- The recipe is FROZEN. Any change (more PLMs, label semantics, scale, seeds) must beat the current numbers
  on SELECT before being sealed on 7401.
