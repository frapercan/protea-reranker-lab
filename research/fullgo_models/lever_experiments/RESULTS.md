# Native reranker PK SCORE-LEVER experiments (OFFLINE, validation frame)

Two score-levers on the native reranker's PK booster, measured against the baseline
PK number on the VALIDATION frame (220->227), with the EXACT baseline recipe
(no TOI, no PK-known exclude; cafaeval ia/prop=fill/norm=cafa/no_orphans/th_step=0.01;
OBO+IA from lafa_t0_Sep_2025). Goal: attack "0.391 is a floor", PK is where we trail.

These validation-frame PK numbers are OPTIMISTIC (boosters tuned/early-stopped on this
eval split, no TOI, no PK-known exclude). They are only meaningful as a SAME-RECIPE
RELATIVE comparison against the baseline 0.4142, which is the point.

## Baseline (comparison point)

Native PK booster, native_boosters_v5: binary objective, lr=0.05, num_leaves=63,
min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
early-stopped on eval (best_iter=769, eval_auc=0.8989).
**PK f_micro_w = 0.4142** (bpo 0.2952 / cco 0.4926 / mfo 0.4549).

## Results

| lever | config | PK f_micro_w | delta vs 0.4142 | verdict |
|-------|--------|--------------|-----------------|---------|
| S2  | scale_pos_weight=45.34 (=#neg/#pos), else identical; early-stop best_iter=369 | **0.4162** | **+0.0020** | helps (small) |
| S13 | boosting_type=dart, drop_rate=0.1, 769 fixed rounds, else identical | ABORTED | n/a | intractable on this box |

PK train rows = 67,555,670 (pos 1,457,807 / neg 66,097,863; scale_pos_weight = 45.341).
PK eval rows  = 6,847,680 (pos 183,187).

## S2 (scale_pos_weight) detail

eval_auc 0.8991 (baseline 0.8989; AUC ~unchanged, as expected -- scale_pos_weight
rescales scores, not the ranking). PK f_micro_w 0.4162, per-ns bpo 0.2955 / cco 0.4911
/ mfo 0.4620. Net +0.0020 over baseline: a small but positive, internally-consistent
lift, concentrated in MFO. Plausible mechanism: upweighting the rare positives shifts
the score distribution so the th_step=0.01 sweep finds a marginally better F operating
point per namespace.

## S13 (DART) detail: ABORTED as intractable

DART (boosting_type=dart, drop_rate=0.1, fixed 769 rounds, all other params = baseline)
on the PK split (67,555,670 train rows x 69 features) did NOT converge in a tractable
wall-clock budget on this box. It was killed after ~210 min of pure training (>3.5 h
wall incl. the 2.5 GB data load) at iteration <769, with no booster ever saved.

Mechanism: DART re-scores the entire growing tree ensemble every round (it drops a
random subset of existing trees and renormalizes), so per-round cost grows superlinearly
with the tree count. On 67.5M rows the late rounds dominate and the fixed 769-round
budget is simply too expensive here. RAM was healthy throughout (~36 GB RSS, >15 GB
avail), so this was a COMPUTE-TIME intractability, not OOM. By contrast S2 (plain GBDT)
early-stopped in ~20 min. No PK f_micro_w exists for S13 because the model never
finished; the run is logged to MLflow as status=ABORTED_INTRACTABLE for completeness.

If DART is to be revisited, it needs either (a) a much smaller round budget (e.g. ~100),
(b) GPU/`device=cuda`, or (c) PK subsampling. As a same-recipe relative comparison on
this hardware, DART is not viable for the full PK split as specified.

## MLflow

Experiment `native-reranker-levers` (id 4) on http://127.0.0.1:5000.
- S2 : run pk-lever-s2 (ffc6405aa1994d198efd3579e7302b08), pk_f_micro_w_validation=0.416212, delta +0.002012.
- S13: run pk-lever-s13 (0a9389c510524fb085cf13d8665c7e71), status=ABORTED_INTRACTABLE (no score metric).

## Files

- train_pk_levers.py  : PK-only trainer (loads v5 train+eval PK rows, one lever per run).
- apply_score_pk.py   : apply a lever PK booster to the validation eval.parquet PK rows + cafaeval (baseline recipe).
- rescore_only.py     : re-score from existing pred/gt TSVs (absolute-path cafaeval fix).
- log_mlflow_levers.py: log a lever run to MLflow.
- s2/, s13/           : per-lever booster (ensemble_gbm_PK.txt), train_meta.json, pred/gt TSVs, result_<lever>.json.
