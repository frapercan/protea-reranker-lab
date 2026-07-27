# Faithful DeepGO-SE vs the deployed two-tower (PK obsolescence test)

Status: DRAFT (training + measurement in progress). Numbers filled by
`storage/deepgose_faithful/run_headtohead.py` -> `headtohead_result.json`.

Goal: build a FAITHFUL DeepGO-SE (correcting the impoverished prior run), plug it into the REAL
offline reranker-lab true-board-frame, and test whether it can OBSOLETE the deployed two-tower
sparse classifier on the PK cells (PK-MFO / PK-BPO / PK-CCO) under the temporal gate.

## 1. Reproduction fidelity

Faithful ELEmbedding base reused verbatim from the proven harness (deepgo2 `base.py`/`models.py`
geometric EL losses: NF1-4 as in `train_gat.py`; `DeepGOModel.forward = MLPBlock+Residual ->
sigmoid(x . (go_embed+hasFunc)^T + |rad|)`; SE ensemble = train N seeds, select top-k by valid BCE,
combine class scores avg/min; min = semantic entailment).

| Faithful element | Impoverished prior run | THIS faithful build |
|---|---|---|
| Protein input | raw ESM2-3B mean 2560-d | **learned k-WTA d8979601 codes 2048-d** (`generator_frames/learned_2048.npy`, aligned to `accs.json`; = `clf_protein_codes_big.npz` basis) |
| Cross-aspect MF->BP | none (BP-only) | **MF model trained first -> MF predictions fed as ADDITIONAL BP input** (`deepgogat_mfpreds_plus`); BP input = concat(k-WTA 2048, mf_preds 776) = 2824-d |
| Axioms | (BP-only harness) | **FULL go-plus-equivalent EL axioms** `go_2025-07-22.norm` (mowl EL-normalized full go.owl; 8 relations incl regulates/part_of/has_part; nf1 70618, nf2 10177, nf3 10173, nf4 17699), NOT is_a+part_of only |
| Aspects | BP only | MF + BP + CC (CC also mfpreds cross-aspect) |
| Pipeline | standalone `se_scores` harness | plugged into the real true-board-frame cafaeval (percut_rerank frame), head-to-head vs deployed two-tower |

DEVIATIONS (stated, forced by env): (a) ELEmbedding base, NOT GAT/DGL -- dgl==1.1.2 is infeasible on
this torch 2.11/cu128 env; the task sanctions the ELEmbedding base + mf-preds-as-input, with the
cross-aspect feeding (the #1 faithful element) intact. (b) embed_dim 1024 (GAT-faithful default) for
the 12 GB GPU + tight disk. (c) 5 SE models, top-3 combine (deepgo2 uses 10, top-6). (d) epochs<=25 +
early stop; el_loss on 16384 sampled axioms/normal-form/step. Strict DeepGO2 `mfpreds` uses mf_preds
ALONE as the BP input; the task wording ("MF predictions as an ADDITIONAL input") -> concat with the
k-WTA codes, retaining the protein representation. [primary build]

## 2. Temporal gate + cross-aspect leakage guard

* Training labels = `generator_frames/labels.npz`, t0-frozen propagated annotations (<= v227 =
  2025-09-04), strictly PRE the v227->v230 eval window. Trainable terms = aspect terms with >=50 t0
  annotations (MF 776 / BP 3905 / CC 631); all other GO classes are axiom-only zero classes (the
  DeepGO-SE route to rare/novel terms).
* Eval targets scored from `eval_protein_codes.npz` k-WTA codes (PK-MF 1086, PK-BP 4402, PK-CC 1571).
* CROSS-ASPECT LEAKAGE GUARD: the MF predictions fed into the BP model are MF-model OUTPUTS trained on
  t0 labels only. No v227->v230 MF label can enter the BP input. (Protein overlap between the 88,212
  training frame and the eval targets is expected and correct: the temporal gate is on LABELS, not
  protein identity; PK targets legitimately carry t0 annotations and are measured only on their
  v227->v230 gains.)

## 3. True-board-frame measurement (identical wrapper for anchor and challenger)

cafaeval call (verbatim from `percut_rerank/score_cafaeval.py`): `cafa_eval(go-basic.obo, pred,
groundtruth_PK.tsv, ia=IA.tsv, no_orphans=True, norm="cafa", prop="fill",
exclude=groundtruth_PK_known.tsv, toi_file=TOI, th_step=0.01)`, metric = `f_micro_w`.

Deployed two-tower anchor (percut-graft), reproduced with MY wrapper on the deployed
`predictions/pk/pk.tsv` (matches recorded `result_9cell.json` within ~0.007, cafaeval build/threshold
grid); a per-protein IA-weighted micro-F decomposition reproduces cafa_eval to 6 decimals (parity
proven), enabling an exact paired protein-bootstrap:

| PK cell | Deployed two-tower anchor (my wrapper) | recorded result_9cell.json |
|---|---|---|
| PK-MFO | 0.24831 | 0.24163 |
| PK-BPO | 0.14351 | 0.14024 |
| PK-CCO | 0.26770 | 0.26554 |

DeepGO-SE is scored with the IDENTICAL wrapper -> frame-internally consistent head-to-head.

## 4. Head-to-head result  [TO FILL after training]

## 5. Cross-aspect effect (BP-mfpreds vs BP-kwta vs impoverished prior)  [TO FILL]

## 6. Two-tower retire verdict  [TO FILL]

Artifacts: `storage/deepgose_faithful/` (train_infer_faithful.py, measure_pk.py, run_headtohead.py,
headtohead_result.json, models_*, se_scores_*). Reproducibility note: the two-tower's learned
ProjHead + SVD/PPMI GO-code basis were not all persisted; this DeepGO-SE is fully reproducible from
the frozen frame + these scripts.
