# Reproduce the 0.349 ensemble (frozen recipe)

All paths are the dev-box absolute paths used during the 2026-06-14 build; adjust as needed. Requires
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

## 3. Train the classifier (6-PLM + ASL) and predict

    $PY train_classifier.py m0v3_data.npz  m0_asl_pred.tsv    # TEST  -> 0.326 standalone
    $PY train_classifier.py sel_data.npz   sel_clf_pred.tsv   # SELECT -> 0.305 (recipe validation)

## 4. Learned ensemble (fit SELECT, seal TEST)

    $PY ensemble_seal.py            # -> NK 0.447 / LK 0.402 / PK 0.199 / MEAN 0.349

## 5. Standalone scoring with the exact harness

    $PY evaluate_exact_harness.py --pred m0_asl_pred.tsv --toi   # classifier alone on 7401

## Notes

- `v227_exp_freq.tsv` / `v220_exp_aspect.tsv` (term frequency for the log_freq feature) are dumped from
  Postgres (see the queries inline in the scripts). `select_pk_known.tsv`, `select_gt_{cat}.tsv` come
  from the platform eval-set download endpoints (eval set a3be0a6d).
- The recipe is FROZEN. Any change (more PLMs, label semantics, scale) must beat the current numbers on
  SELECT before being sealed on 7401.
