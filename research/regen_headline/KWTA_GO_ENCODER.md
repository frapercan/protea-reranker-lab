# Learned k-WTA GO ontology encoder + sequence to GO-code separability probe

Receipt JSON: `storage/regen_headline/KWTA_GO_ENCODER.json`
Model assets + per-phase JSON/logs: `storage/kwta_go_encoder/`
Frozen data only. Co-annotation computed DIRECTLY from `storage/protea-frozen-v227-2025-09-04/reference_annotations.parquet` (v227 = t0); the live DB (`term_cooccurrence`) was never queried. Evaluation is on post-t0 NOVEL terms (gt v230 minus v227 known); every training signal (co-annotation, structure, text, protein true terms) is strictly <= t0.

## One-line verdict
The learned k-WTA GO encoder is a strong, REPRODUCIBLE representation asset (intrinsic term-term separation AUC **0.84** vs the fixed-recipe two-tower's **0.21**), but it is **NOT a BP lever**: sequence to GO-code sparse-overlap neither out-separates the dense annotation-RAG at the protein tail nor converts to f_micro_w (every true-frame delta negative; the learned score is no better than a random control). The BP wall is protein-to-term SEPARABILITY, not term-representation form.

---

## Deliverable 1 - the representation asset (POSITIVE)

A learned, end-to-end k-WTA GO encoder over term-feature blocks (co-annotation PPMI SVD256 + ontology-structure SVD128 + BioBERT text 768), MLP to a 2048-d ReLU + k-WTA(128) sparse code, trained with a multi-positive InfoNCE + margin objective whose HARD NEGATIVES are text/ontology-close but never-co-annotated term pairs. Deterministic (seed 0). Everything saved: `encoder_*.pt`, `go_codes_*.npz` (25,699 BP terms), and `bases.npz` (the co-annotation + structure SVD components, text whitening mean, ancestor columns) - this closes the two-tower's unsaved-SVD-basis reproducibility gap.

Intrinsic quality (held-out pairs, seed 123; does the hard-negative objective work?):

| code | pos overlap (co-occurring) | hard-neg overlap (text-close, non-co-occurring) | separation AUC |
|---|---|---|---|
| **learned (coann+struct+text)** | 0.462 | 0.152 | **0.841** |
| learned (coann only) | 0.461 | 0.159 | 0.836 |
| **fixed-recipe two-tower** | 0.234 | **0.404** | **0.209** |

The learned code gives co-occurring / same-process terms OVERLAPPING codes and pushes text-close-but-non-co-occurring terms APART - the hard-negative objective working. The fixed-recipe two-tower does the OPPOSITE (hard negs overlap MORE than positives, AUC below random): its codes are text-dominated, conflating text similarity with function. Smoke pairs confirm: glycolysis~gluconeogenesis 0.72 (co-occurring, high), glycolysis~translation 0.03 and apoptosis~cell_cycle 0.0 (not co-occurring, low), DNA_repair~DNA_replication 0.71.

Ablation - which term-feature block matters (val separation AUC): **co-annotation is the CORE (0.846 alone)**; +structure adds +0.004 (0.850); +text is neutral (0.850). Text does not help separability because text is precisely what defines the hard negatives - it cannot also separate them. This confirms the author's premise that co-annotation is the core signal.

## Deliverable 2 - the BP separability probe (NEGATIVE)

Sequence to GO-code projection (`seq_head_*.pt`): the deployed Ankh/d8979601 protein k-WTA code (2048-d) projected through an MLP + k-WTA(128) into the frozen GO-code space, trained full-BP-vocab IA-weighted so the projected code sparse-overlaps its true t0 terms. The fixed-recipe two-tower is carried through the IDENTICAL head recipe as a controlled comparison (isolating the GO representation).

Per-protein tail separability (IA>=4, novel-eligible; the crux vs the dense annotation-RAG that was flat):

| cell | learned k-WTA tail AUC | dense annotation-RAG tail AUC | Q_TAX ratio (learned / dense / classifier) |
|---|---|---|---|
| PK_BP | 0.726 | 0.725 | 0.0026 / 0.0036 / 0.019 |
| LK_BP | 0.818 | 0.829 | 0.0009 / 0.0025 / (n/a) |

The learned sparse-overlap does **not** separate the BP tail better than the dense baseline - tail AUC is tied (PK) or slightly behind (LK), and the marginal-over-kNN true:false ratio stays flat and near-zero, far below the classifier. Intrinsic term-code quality (0.84 vs 0.21) does NOT transfer to protein-to-tail-term separability: the wall sits at protein-to-term assignment, not term representation.

f_micro_w, TRUE board frame (prop=fill, norm=cafa, no_orphans, toi; PK -known per evaluation.nf:279), extras quantile-calibrated onto the pool score distribution identically per arm; cafa_eval under the PROTEA venv. Noise floor 0.0034.

| cell (anchor A) | arm | delta top100 | delta top200 |
|---|---|---|---|
| **PK_BP** (0.11033) | LEARNED | -0.05876 | -0.07237 |
| | FIXEDSCORE control (random) | -0.06283 | -0.07631 |
| | two-tower (fixed) | -0.01804 | -0.03466 |
| | dense annotation-RAG | -0.03190 | -0.05352 |
| **LK_BP** (0.29896) | LEARNED | -0.21548 | -0.24555 |
| | FIXEDSCORE control (random) | -0.22695 | -0.25397 |
| | two-tower (fixed) | -0.09826 | -0.16205 |
| | dense annotation-RAG | -0.13000 | -0.18959 |

Every arm is strongly NEGATIVE - the generation channel does not convert, in the true frame, under the temporal gate. Two disciplined reads:
- **Learned approximately equals the random FIXEDSCORE control** (PK -0.059 vs -0.063; LK -0.215 vs -0.227): the learned overlap score carries essentially no board-usable signal over volume in ranking its own extras. The fixed-score control is what makes this trustworthy.
- **Recall vs ordering:** LEARNED is the MOST negative because it is the MOST generative (178k extras/top100-PK vs the two-tower's 129k) - it REACHES more tail but cannot ORDER true above false, so each false extra forfeits its ancestor's inheritance under prop=fill. This is reach without separability = net harm. The wall is SEPARABILITY, not reachability.

## Bottom line
A genuinely-learned k-WTA GO representation with separability hard negatives is a clean, reproducible ASSET that decisively out-separates the fixed two-tower recipe at the term level and fixes its reproducibility gap. It does NOT move the BP wall: it does not out-separate the dense annotation-RAG at the protein tail and does not convert to f_micro_w. Consistent with the campaign's converged picture - the deployed reranker is near-optimal and the only channel with signal is generation, whose payoff at the board is approximately zero because the BP tail is genuinely separability-limited, not representation-limited.
