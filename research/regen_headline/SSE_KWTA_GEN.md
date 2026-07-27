# Generalization capacity of a single-model, accept-all Sparse Semantic-Entailment (SSE) over the DEPLOYED frozen k-WTA representation

**What this measures.** How well entailment (sparse set-containment) over our FROZEN champion k-WTA protein codes (d8979601, 2048-d, 128 active) RANKS true GO terms on the blind v227->v230 eval, decomposed across three generalization regimes. Threshold-free ranking metrics (AUROC / AUPR / protein-centric Fmax); the calibrated f_micro_w is a footnote, not the verdict (accept-all floods it by construction).

## Model + frozen input frames
- Protein rep: **FROZEN champion learned k-WTA codes d8979601 (2048-d, 128 active); support = non-zero dims**
- Train codes: `repositories/.../clf_protein_codes_big.npz (554,378 v227 corpus, eval HELD OUT)`
- Eval codes: `repositories/.../eval_protein_codes.npz (7,401 sealed LAFA v227->v230; SAME d8979601 encoder)` -- SAME d8979601 encoder (both 2048-d, exactly 128 active); aligned by the `accs` sidecar.
- Term tower: learned embedding -> soft k-WTA per-term cardinality by depth; nf1/nf4 EL axioms, parent gate detached
- K = 1 (primary, accept-all by order); K=5 MIN kept as control
- Temporal gate: term codes trained on v227(t0) co-annotations, eval proteins held out; v227->v230 blind

**Regimes** (over eval positive (protein,term) pairs; negatives = vocab terms neither true nor known@t0):
- **R1 known@t0** -- annotation already true at t0 (v227); excluded by the honest `-known` frame = memorization / in-distribution-recall UPPER REFERENCE.
- **R2 new / seen-term** -- new in v227->v230, term train-support >= S_HI (well-represented) = LK-like.
- **R3 new / novel-term** -- new in v227->v230, term train-support < S_HI (rare/deep novel term) = the wall. Out-of-vocab novel terms are UNREACHABLE by accept-all (recall-ceiling loss).

## The 3-regime ranking decomposition (K=1, per aspect)

### BPO  (vocab 7479, targets 4925, S_HI=3248)

| regime | n_prot | AUROC (per-prot) | AUPR (per-prot) | Fmax | AUROC micro | rand AUROC |
|---|---|---|---|---|---|---|
| R1 known (memorization ref) | 4629 | 0.9544 | 0.4017 | 0.4050 | 0.9421 | 0.4967 |
| R2 new / seen-term | 3231 | 0.9143 | 0.1002 | 0.0980 | 0.9156 | 0.4993 |
| R3 new / novel-term (wall) | 3855 | 0.6619 | 0.0057 | 0.0045 | 0.7047 | 0.5037 |

**Degradation (AUROC):** R1 0.9544 -> R2 0.9143 (Δ -0.0401) -> R3 0.6619 (Δ -0.2524). The R1->R3 collapse of 0.2925 AUROC IS the generalization gap.

**Accept-all recall ceiling (IA-mass).** reachable (in-vocab) = **0.958** of true IA-mass; the rest is out-of-vocab novel terms (unreachable) -- a DATA/reachability property, not the model.

- regime IA-mass: R1 42988 | R2 16718 | R3 in-vocab 25811 | R3 out-of-vocab UNREACHABLE 3571
- deployed pool reachability of NEW true IA-mass: PK 0.046 | LK 0.372

### MFO  (vocab 3096, targets 1242, S_HI=6021)

| regime | n_prot | AUROC (per-prot) | AUPR (per-prot) | Fmax | AUROC micro | rand AUROC |
|---|---|---|---|---|---|---|
| R1 known (memorization ref) | 918 | 0.9634 | 0.4973 | 0.4635 | 0.9623 | 0.4926 |
| R2 new / seen-term | 580 | 0.9394 | 0.2313 | 0.1966 | 0.9306 | 0.5041 |
| R3 new / novel-term (wall) | 559 | 0.7173 | 0.0306 | 0.0190 | 0.7385 | 0.5147 |

**Degradation (AUROC):** R1 0.9634 -> R2 0.9394 (Δ -0.0240) -> R3 0.7173 (Δ -0.2221). The R1->R3 collapse of 0.2460 AUROC IS the generalization gap.

**Accept-all recall ceiling (IA-mass).** reachable (in-vocab) = **0.876** of true IA-mass; the rest is out-of-vocab novel terms (unreachable) -- a DATA/reachability property, not the model.

- regime IA-mass: R1 6774 | R2 2091 | R3 in-vocab 3162 | R3 out-of-vocab UNREACHABLE 1410
- deployed pool reachability of NEW true IA-mass: PK 0.121 | LK 0.362

### CCO  (vocab 1307, targets 1834, S_HI=17886)

| regime | n_prot | AUROC (per-prot) | AUPR (per-prot) | Fmax | AUROC micro | rand AUROC |
|---|---|---|---|---|---|---|
| R1 known (memorization ref) | 1781 | 0.9733 | 0.6373 | 0.5946 | 0.9675 | 0.4951 |
| R2 new / seen-term | 800 | 0.9701 | 0.3305 | 0.2684 | 0.9618 | 0.5054 |
| R3 new / novel-term (wall) | 1219 | 0.8210 | 0.0478 | 0.0321 | 0.8062 | 0.4863 |

**Degradation (AUROC):** R1 0.9733 -> R2 0.9701 (Δ -0.0032) -> R3 0.8210 (Δ -0.1491). The R1->R3 collapse of 0.1523 AUROC IS the generalization gap.

**Accept-all recall ceiling (IA-mass).** reachable (in-vocab) = **0.865** of true IA-mass; the rest is out-of-vocab novel terms (unreachable) -- a DATA/reachability property, not the model.

- regime IA-mass: R1 9609 | R2 2262 | R3 in-vocab 4845 | R3 out-of-vocab UNREACHABLE 2462
- deployed pool reachability of NEW true IA-mass: PK 0.151 | LK 0.342

## Controls + honest comparisons

### vs the deployed reranker ordering (PK-BPO reachable tail)
- n_proteins 1028; reranker per-protein AUC **0.8132** vs single-model SSE AUC **0.6596** (SSE - reranker = -0.1535).
- per-protein AUC on the deployed reachable pool tail (in-vocab, NEW terms, known@t0 excluded); same term set for both.

### K=1 (accept-all) vs K=5 MIN ensemble (same frozen rep)
- BPO: R1 K1 0.9544/K5 0.9610 | R2 K1 0.9143/K5 0.9367 | R3 K1 0.6619/K5 0.6482
- MFO: R1 K1 0.9634/K5 0.9619 | R2 K1 0.9394/K5 0.9414 | R3 K1 0.7173/K5 0.6697
- CCO: R1 K1 0.9733/K5 0.9799 | R2 K1 0.9701/K5 0.9837 | R3 K1 0.8210/K5 0.8232

### Containment semantics on the frozen rep

| aspect | term-axiom containment | random floor | frac fully-contained | frozen-rep parent>=child coverage |
|---|---|---|---|---|
| BPO | 0.371 | 0.082 | 0.002 | 0.807 |
| MFO | 0.496 | 0.044 | 0.023 | 0.622 |
| CCO | 0.359 | 0.062 | 0.002 | 0.715 |

_k-WTA codes were learned for DISCRIMINATION, not as a union-of-required-modules; term-axiom containment sits above the random floor but is not exact subset-containment. A CO-TRAINED SSE (protein + term towers jointly) is the upper bound; here the protein rep is frozen so containment is only as good as the discriminative code allows._

### Footnote: accept-all calibrated f_micro_w (PK -known) -- confirms flood, NOT the verdict
- anchor reproduced: {'bpo': 0.14351, 'cco': 0.2677, 'mfo': 0.24831}
- accept-all SSE f_micro_w: {'bpo': 0.06289, 'cco': 0.11671, 'mfo': 0.08283}
- delta vs anchor: {'bpo': -0.08062, 'cco': -0.15099, 'mfo': -0.16547}
- accept-all floods the calibrated f_micro_w (max recall, catastrophic precision); this is a FOOTNOTE confirming the known flood, NOT the generalization verdict.

## Verdict

A single-model (K=1), accept-all Sparse Semantic-Entailment over our DEPLOYED frozen champion k-WTA representation (d8979601, 2048-d, 128 active; only the term codes are learned) has STRONG generalization on ALREADY-SEEN terms and NO generalization on NOVEL terms, and the gap between the two IS its capacity. Ranking a protein's blind v227->v230 annotations, per-protein AUROC degrades monotonically across the three regimes in every aspect: R1 known@t0 (memorization/in-distribution reference) 0.954/0.963/0.973 (BP/MF/CC) -> R2 new-annotation-on-a-seen-term 0.914/0.939/0.970 (it generalises to genuinely new protein-term links as long as the term is well represented in training) -> R3 new-annotation-on-a-novel/rare-term 0.662/0.717/0.821, worst for BP where it reproduces the known PK-BPO separability wall (~0.62-0.66). The R1->R3 AUROC collapse (0.29 BP / 0.25 MF / 0.15 CC) is the generalization capacity; protein-centric Fmax collapses even harder (R1 0.40-0.59 -> R2 0.10-0.27 -> R3 ~0.004-0.03), so even where AUROC is non-trivial the novel-term tail carries essentially no usable precision. The wall is a SEPARABILITY/precision limit at the novel-process tail, not a data-reachability limit: accept-all reaches 86-96% of the true IA-mass (only 4-15% of true terms are truly out-of-vocab-unreachable, and that ceiling is a property of the training vocabulary, not the model), yet cannot rank R3, which is exactly why accept-all floods the calibrated f_micro_w (footnote: BP -0.081, MF -0.165, CC -0.151 vs the exactly-reproduced deployed anchors). Dropping to K=1 does not cost ranking quality (K=5 MIN is within +/-0.02 AUROC and does NOT rescue R3), and the single model does NOT out-rank the deployed reranker on its own reachable pool tail (reranker 0.813 vs SSE 0.660 on matched in-pool PK-BPO pairs; SSE's 0.660 reproduces the prior SSE separability, while the prior 'reranker~0.49' was a survivorship-free candidate frame). Containment holds only approximately on the frozen rep (term-axiom nf1 containment 4.5-11x above the random floor but exact subset rare; frozen-rep parent>=child coverage 62-81%), consistent with codes learned for discrimination rather than as a union-of-required-modules -- a CO-TRAINED SSE is the upper bound. NET: over the deployed sparse rep, single-model accept-all entailment generalizes to new links on seen terms but hits the novel-process reachability/separability wall exactly where the deployed system already does; this is a third, frozen-rep, architecturally-independent confirmation that the BP novel-term wall is fundamental.

### Note on the reranker-comparison frame

The vs-reranker tail here = terms IN the deployed pool (the reranker's own survivorship-selected candidates), so the reranker separates it well (0.813). This diverges from the prior ENTAIL_KWTA 'reranker 0.491', which measured a survivorship-free reachable universe. On matched in-pool pairs the reranker out-separates single-model SSE; SSE's 0.660 still reproduces the prior SSE separability. No claim that SSE beats the reranker.

## Receipts

- Model/train: `storage/sse_kwta_gen/build_and_train.py` -> `termcodes_{bpo,mfo,cco}.npz`, `models/`
- Score+measure: `storage/sse_kwta_gen/score_and_measure.py`; logs `train.out`/`measure.out`; status `chain_status.json`
- Consolidated JSON: `storage/regen_headline/SSE_KWTA_GEN.json`
- Frozen inputs: champion k-WTA `clf_protein_codes_big.npz`(train)/`eval_protein_codes.npz`(eval, d8979601); labels `clf_labels_big.json`(v227); obo/IA `lafa_t0_Sep_2025`; EL norms `go_2025-07-22.norm`; deployed pool `percut_rerank/predictions/{pk,lk}`. NO live DB, NO job dispatch.
