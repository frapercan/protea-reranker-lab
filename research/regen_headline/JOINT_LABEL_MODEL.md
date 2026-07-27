# The BELIEVABLE experiment: joint label-channel model (TransFew architecture, our backbone)

**Question.** Our five short generation probes each bolted ONE channel onto the deployed pipeline,
cold, and all NO-GO'd. TransFew wins the two lost BP cells by learning sequence + GO-text +
GO-hierarchy JOINTLY. Does a jointly-trained model in TransFew's architecture, on our validated
Ankh backbone, beat our bolt-ons on LK-BPO and PK-BPO. If yes, our negatives were a methodology
artefact. If no, the frontier is real.

**Verdict: NO.** Joint end-to-end training of the label channel does not beat the deployed system
on either lost BP cell. It does not even reach the deployed baseline. LK-BPO -0.028, PK-BPO -0.019
in the true frame under the temporal gate. The bolt-on negatives were NOT a methodology artefact.

## The model (faithful replication, our compute)

- **Protein tower**: MLP over the validated Ankh k-WTA code (`d8979601`, 2048-d, the deployed encoder).
- **Label tower**: BioBERT GO-text embedding (768-d) refined by a 2-layer GCN over the GO DAG
  (is_a + part_of, symmetric-normalized, 124,225 edges over 25,699 BP terms). This is TransFew's
  stated mechanism for rare terms: a tail term's row is pulled toward its DAG neighbours.
- **Fusion**: cross-attention, the protein rep queries the label bank, residual-fused.
- **Output**: label-embedding layer `s(protein, term) = (fused . label) / sqrt(d) + bias`.
- **Loss**: ASL (asymmetric focal), end-to-end. 2.26M params, trains in ~21 min on the RTX 3060.
- The one difference from our bolt-ons and from the two-tower: everything (protein MLP, GCN label
  tower, cross-attention) is trained jointly against the protein's BP annotation set.

Diagnostic note kept as a receipt: unscaled dot-product logits over 512 dims saturate the sigmoid
and the gradient vanishes (measured 0.000 on real data). Scaling by 1/sqrt(d) plus a
class-imbalance bias prior unblocks training. Any "joint model fails" claim without this fix would
have been a training artefact, not a frontier.

## Discipline (all gates held)

- **Temporal-transfer gate**: train on snapshots <= v220-v225, early-stop on v225-v227, test blind
  on v227-v230. Best val masked-AP 0.457 (the model genuinely ranks newly-gained BP terms in a
  held-out temporal window, so it learned real signal, this is not a dead model).
- **Leakage, exact**: the 7,401 eval targets are held OUT of both training and early-stop
  (disjoint split, as TransFew trains). Raw gained (protein,term) pairs seen in training labels:
  **0 / 8,917 (LK), 0 / 41,727 (PK)**. Removes the IEA-vs-experimental ambiguity entirely.
- **Decider**: f_micro_w in the true board frame (obo+IA, prop=fill, norm=cafa, no_orphans, toi;
  PK adds -known). Harness validated: it reproduces the lab's own `result_9cell.json` deployed
  cells (LK-BPO 0.311, PK-BPO 0.140) to within 0.002. Never ranked by AUC.

## Result, one-variable framing (all arms, one harness)

| arm | LK-BPO | PK-BPO |
|---|---|---|
| **deployed reranker** (baseline to beat, this harness) | **0.313** | **0.144** |
| joint, ORDERING (rescore the SAME pool candidates) | 0.280 | 0.111 |
| joint, GENERATION (top-K over full BP vocab, best K) | 0.285 | 0.124 |
| joint best (max of ordering / generation) | 0.285 | 0.124 |
| **delta vs deployed** | **-0.028** | **-0.019** |
| bolt-on sum (as features ON deployed): GO-text +0.012 (LK), classifier +0.022 (PK, small-k) | > joint | > joint |
| internal board (TransFew, this true frame, result_9cell.json) | 0.348 | 0.117 |
| TransFew published (paper frame, not this frame) | 0.512 | 0.294 |

Note on the board number: in this true frame the deployed system already BEATS the internal board on
PK-BPO (0.144 vs 0.117) and trails on LK-BPO (0.313 vs 0.348). The paper-frame 0.512 / 0.294 are a
different metric frame and are not comparable line-for-line to the true-frame cells here.

## Recall vs ordering (the question the bolt-on probes could not answer)

- **ORDERING is a LOSS.** Rescoring the exact deployed pool candidates, the joint model is a WORSE
  ranker than the deployed reranker (-0.032 LK, -0.033 PK). The deployed reranker, which uses the
  kNN neighbour feature stack, orders the existing candidates better than the joint label model.
- **GENERATION is REAL but does not convert.** Over the full BP vocab the joint model proposes
  genuine new tail terms our pool lacks:
  - LK: 990 new-true / 13,287 proposed = **7.45%** precision (top50), added-true IA-mass 1,034.
  - PK: 9,979 new-true / 130,585 proposed = **7.64%** precision (top50), added-true IA-mass 10,039.
  This sits between the 1% co-occurrence bar and the 11.59% classifier bar, comparable to the 5.8%
  network bar. So the label channel DOES know where some missing branches are (5-8x the co-occ bar).
  But it does not convert: generation only edges ordering (PK 0.124 vs 0.111) and stays below
  deployed, because under prop=fill the ~92% false extras forfeit their ancestors' free inheritance.
  This is the same SEPARABILITY wall the bolt-on probes hit, now confirmed for a jointly-trained
  label channel: the tail candidates are simply not separable from false ones at the metric's cut.

## One-line verdict

Joint training of the label channel, in TransFew's architecture on our validated Ankh backbone,
does NOT beat our bolt-on methodology and closes NONE of the gap: it loses LK-BPO by 0.028 and
PK-BPO by 0.019 in the true frame under a clean temporal gate with zero leakage. The label channel's
only signal is GENERATION (real tail terms at ~7.6% precision) and it does not survive the metric,
exactly as the bolt-ons found. The BP-tail wall is separability, not an artefact of testing the
channels cold. The frontier is real.

Receipts: `storage/joint_model/joint_result.json` (= `storage/regen_headline/JOINT_LABEL_MODEL.json`),
model `storage/joint_model/joint_model.pt`, code `storage/joint_model/train_joint.py`,
harness `storage/joint_model/trueframe.py`, log `storage/joint_model/train.log`.
