# Transition-separability probe: can a learned known-annotation-conditioned term-transition score beat the deployed reranker on LK-BPO and PK-BPO?

**Verdict: NO-GO.** A learned known-conditioned score does **not** produce a real, deployable
true-frame cafaeval gain on either lost cell. The +0.072 (LK-BPO) / +0.076 (PK-BPO) gaps to
TransFew are **not** closed; the honest deployable delta is slightly **negative** in both cells
(PK-BPO -0.0059, LK-BPO -0.0088). The literal TransFew mechanism (higher-order co-occurrence
diffusion) is doubly dead: its score is **below random** on its own.

All numbers are frozen-data only. No live DB, no job dispatch. cafaeval frame reproduced exactly:
lab obo+IA (`lafa_t0_Sep_2025`), `-prop fill -norm cafa -no_orphans -toi groundtruth_terms_of_interest.txt`,
and `-known groundtruth_PK_known.tsv` for PK. Arm A reproduces the deployed cell to the digit
(PK 0.1402366, LK 0.3109618 — identical to `result_9cell.json`).

---

## Phase 1 — separability ceiling (per-protein AUC, grouped-CV by protein, leakage-clean)

| feature / model (per-protein AUC)        | LK-BPO | PK-BPO |
|------------------------------------------|:------:|:------:|
| deployed reranker (incumbent)            | 0.835  | **0.505** (chance) |
| first-order `association_cross`          | 0.751  | 0.640  |
| learned GBDT, known-conditioned feats    | 0.829  | 0.788  |
| learned logistic, known-conditioned      | 0.797  | 0.648  |
| **higher-order transition, 1-hop PPMI, ALONE**  | **0.409** | **0.430** |
| **higher-order transition, 2-hop diffusion, ALONE** | **0.459** | **0.408** |
| GBDT + transition feature                | 0.861  | 0.820  |

Two hard readings:
1. **The higher-order transition (the literal TransFew label-diffusion mechanism) is below random
   on its own** in both cells and never beats first-order `association_cross`. The whole
   "higher-order beats first-order" hypothesis is falsified at the AUC stage. Any lift it shows
   inside the GBDT is a small residual interaction, not a clean separator.
2. **PK-BPO: the deployed reranker ranks at chance (0.505)** while a leakage-clean GBDT over
   known-conditioned features reaches 0.788 — flat across every known-count stratum (reranker
   0.45–0.53 everywhere, GBDT 0.76–0.82 everywhere). This quantifies the "misordering where the
   pool is rich" prediction. LK-BPO is already well-ordered by the deployed reranker (0.835).

## Phase 2 — translate to true-frame f_micro_w, IN-WINDOW ceiling (grouped-CV OOF on the eval window)

Rescoring the **deployed candidate set** (no new candidates), best of {additive weight grid,
full re-rank, rank-average}, both with and without the transition feature:

| cell | incumbent | best blend (arm) | delta | bootstrap CI95 (2000×, paired proteins) | frac_pos |
|------|:---------:|:----------------:|:-----:|:---------------------------------------:|:--------:|
| PK-BPO | 0.14024 | **0.19387** (gbdt+trans, +w1.0) | **+0.0536** | [+0.0440, +0.0633] | 1.000 |
| LK-BPO | 0.31096 | **0.34457** (gbdt+trans, +w1.0) | **+0.0336** | [+0.0208, +0.0459] | 1.000 |

`gbdt_known` **without** the transition feature gives essentially the same gain (PK +0.0517),
so the transition feature is not the source. The bootstrap reimplements cafaeval's IA-weighted
micro-F per protein and matches the point estimate to 5 decimals (`parity_ok=True`).

This looks like a large, significant win. **It is not deployable.** See Phase 3.

## Phase 3 — adversarial controls + the deciding deployable test

### (a) Ordering vs scale/volume controls — the in-window gain IS real ordering
| control | PK-BPO | LK-BPO | reading |
|---------|:------:|:------:|---------|
| FIXED-SCORE constant on BP candidates | 0.030 | 0.075 | collapses far below incumbent → not a submit-everything/volume artifact |
| PERM (blend's exact score multiset, ordering shuffled) | 0.040 | 0.075 | collapses → the ordering carries it |
| SCALEXFER (deployed ordering + blend's score distribution) | 0.1397 (≈inc) | 0.3109 (≈inc) | rescaling the deployed ordering recovers **nothing** → not the cafaeval scale artifact |

So the in-window gain is genuinely the model's **ordering** of tail candidates, not scale/volume.

### (b) DEPLOYABLE temporal test — train on the PAST only (v160–v225, early-stop v225–v227), apply to the blind eval window
Same 12 known-conditioned features as the `gbdt_known` arm above; mirrors how the deployed reranker
is trained. This removes the eval-window label information that grouped-CV leaks.

| cell | val ROC-AUC | eval per-protein AUC | best deployable f | **delta vs incumbent** |
|------|:-----------:|:--------------------:|:-----------------:|:----------------------:|
| PK-BPO | 0.899 | **0.757** (vs deployed 0.505) | 0.13436 | **-0.0059** |
| LK-BPO | 0.866 | 0.815 (vs deployed 0.835) | 0.30213 | **-0.0088** |

Every weight and the full re-rank are negative in both cells.

**This is the decider.** The deployable model keeps almost all of the per-protein ranking lift
(PK AUC 0.757, nearly the 0.82 of the in-window model) yet **loses** f_micro_w. The +0.054 / +0.034
in-window ceiling was therefore **calibration to the eval-window label distribution**, which is
unavailable at deployment — not transferable signal.

## Mechanism (why AUC transfers but f_micro_w does not)

`f_micro_w` is fixed by a single **global** IA-weighted threshold over the pooled prediction. The
known-conditioned model's advantage is two-part: (i) within-protein ranking, which transfers
temporally (per-protein AUC survives train-on-past), and (ii) cross-protein **score calibration**
matched to the eval period's base rates, which does **not** transfer. cafaeval's global-tau,
IA-weighted micro pooling is driven by (ii), so per-protein AUC — the thing that transfers — is the
wrong currency. This is the concrete instance of the standing memo "AUC ranked levers 4× opposite
to f_micro_w" and "lab deltas do not transfer to the board." A fixed-score/scale control catches
the scale artifact; only a **temporal train/test split** catches this calibration artifact.

## Bottom line (go/no-go, with the numbers)

- Deployable true-frame delta: **PK-BPO -0.0059, LK-BPO -0.0088** → does not close the
  **+0.076 / +0.072** gaps to TransFew; it moves the wrong way.
- The literal higher-order transition score is **below random** (AUC 0.41–0.46) and dead on arrival.
- The only positive (in-window ceiling +0.054 / +0.034, CI-clean, controls-clean) is a
  **train-on-test-window artifact** and must not be reported as a result.

**NO-GO** on a learned known-annotation-conditioned term-transition score as a reranker feature for
LK-BPO / PK-BPO. Consistent with the campaign's converged picture: generation, not rescoring, is the
only channel with transferable signal.

## Receipts (all under `storage/regen_headline/`)
- `phase1_separability.json` + `phase1_separability.py` — per-protein AUC table, stratified.
- `phase2_cafaeval.json` + `phase2_cafaeval.py` — in-window true-frame f_micro_w per arm.
- `phase3_controls.json` + `phase3_controls.py` — constant / PERM / SCALEXFER controls.
- `temporal_deployable.json` + `temporal_deployable.py` — **the deciding deployable test**.
- `bootstrap_ci.json` + `bootstrap_ci.py` — exact-parity paired protein bootstrap (parity_ok=True).
- `phase1_scores_{lk,pk}.parquet` — per-candidate OOF model scores.
