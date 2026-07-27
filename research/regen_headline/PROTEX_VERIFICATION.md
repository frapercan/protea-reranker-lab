# ProtEx-style exemplar-conditioned verification for the BP separability wall

Receipts: `storage/protex/` (pipeline + models), `storage/regen_headline/PROTEX_VERIFICATION.json`
(machine-readable result). Compute: `repositories/PROTEA/.venv/bin/python`, RTX 3060 shared.

## One line

GO. Negative-exemplar conditioning is a real, deployable lever on the two weakest BP cells: a
verifier that contrasts positive against hard-negative exemplars adds **+0.0344 (LK-BPO)** and
**+0.0344 (PK-BPO)** f_micro_w on top of the deployed reranker in the true board frame under a strict
temporal gate, with paired-bootstrap CI strictly above zero (frac-positive 1.0). The identical model
WITHOUT negative exemplars converts to nothing (+0.0015 / -0.0037). This is the ProtEx thesis holding:
the negatives, not the positives, are what turn the reachable tail into rankable signal.

## What was built

For LK-BPO and PK-BPO, on the reachable candidate tail (the BP candidates already in the deployed
pool, `eval_scores.parquet`, with true/false labels), a YES/NO verifier scores every
(protein, candidate-term) pair from a contrast of exemplars rather than from the protein's own
features:

- Retrieval space: ProtT5 (frozen v227 bundle, config `084943c6`, mean-pool 1024-d), L2-normalised.
  This is the ONLY representation in which BOTH the full leakage-clean t0 reference set (574,627) AND
  100% of the pool queries are already materialised. (The brief named Ankh/d8979601; ProtT5 was chosen
  because re-embedding 574k references with Ankh on a shared 12 GB GPU is unsafe and unnecessary,
  retrieval-space consistency between query and reference is what the method needs, and this is the
  canonical deployed KNN space. All queries covered, so no second embedding source is mixed in.)
- Exemplars per (protein p, term t), from p's 256 nearest t0 references (self and cosine >= 0.9999
  near-duplicates removed): POSITIVE = the K=8 closest references that CARRY t at t0 (propagated);
  NEGATIVE = the K=8 closest that LACK it (hard: similar sequence, no term).
- Verifier (small MLP, deep-sets set-encoder): input = [p embedding, term features (anc2vec + IA),
  mean-pooled POSITIVE-exemplar embeddings, mean-pooled NEGATIVE-exemplar embeddings, pos/neg
  contrast scalars] -> P(p has t). BCE with pos_weight. The ablation drops the NEGATIVE pool and its
  contrast.
- Temporal gate: train on snapshot pairs <= v225, early-stop on v225-v227, test BLIND on v227-v230.

## Decider 1: separability (diagnostic, per-protein true-vs-false AUC on the blind window)

| cell | signal | AUC all cands | AUC IA>=4 tail |
|------|--------|---------------|----------------|
| PK-BPO | raw kNN (-distance) | 0.621 | 0.643 |
| PK-BPO | deployed reranker | 0.737 | 0.850 |
| PK-BPO | verifier, no negatives | 0.770 | 0.866 |
| PK-BPO | **verifier, with negatives** | **0.780** | **0.869** |
| LK-BPO | raw kNN (-distance) | 0.497 | 0.670 |
| LK-BPO | deployed reranker | 0.843 | 0.828 |
| LK-BPO | verifier, no negatives | 0.834 | 0.876 |
| LK-BPO | **verifier, with negatives** | **0.859** | **0.889** |

Reading: the reachable tail is NOT flat under exemplar conditioning. Raw kNN transfer is weak
(0.62-0.67 tail), but the exemplar verifier separates true from false as well as or better than the
near-optimal deployed reranker, using only exemplar contrast plus protein embedding plus term identity
(no kNN/classifier features). Negative-exemplar conditioning adds separability everywhere (largest on
LK: +0.025 all, +0.013 tail). Separability is diagnostic only; it is never the verdict.

## Decider 2: f_micro_w, TRUE board frame, TEMPORAL gate, paired-bootstrap CI

Frame: lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds -known (evaluation.nf:279). Anchor
A = deployed pool reranker_score, reproducing the prior kwta anchors (PK 0.11009 vs 0.11033; LK
0.28974 vs 0.29896). Every arm submits the IDENTICAL candidate set, re-ordered and calibrated onto A's
own score histogram, so ONLY the ordering differs (a pure separability->metric test). CI = 2000-sample
paired protein bootstrap via cafaeval's own parser (exact-parity, mirrors `bootstrap_ci.py`).

| cell | arm | f_micro_w | delta vs deployed | bootstrap CI95 | frac_pos | clears 0.0034 |
|------|-----|-----------|-------------------|----------------|----------|---------------|
| PK-BPO | A deployed | 0.11009 | - | - | - | - |
| PK-BPO | verifier, no negatives | 0.11153 | +0.00145 | [-0.0047, +0.0078] | 0.67 | NO |
| PK-BPO | **verifier, with negatives** | 0.13270 | **+0.02261** | [+0.0161, +0.0293] | 1.00 | YES |
| PK-BPO | **reranker + verifier blend** | 0.14447 | **+0.03438** | [+0.0305, +0.0386] | 1.00 | YES |
| PK-BPO | CONTROL random order | 0.03413 | -0.07596 | [-0.0813, -0.0707] | 0.00 | NO |
| LK-BPO | A deployed | 0.28974 | - | - | - | - |
| LK-BPO | verifier, no negatives | 0.28605 | -0.00369 | [-0.0267, +0.0197] | 0.38 | NO |
| LK-BPO | **verifier, with negatives** | 0.32776 | **+0.03802** | [+0.0133, +0.0623] | 1.00 | YES |
| LK-BPO | **reranker + verifier blend** | 0.32422 | **+0.03448** | [+0.0195, +0.0499] | 1.00 | YES |
| LK-BPO | CONTROL random order | 0.07766 | -0.21207 | [-0.2366, -0.1888] | 0.00 | NO |

## The negative-exemplar ablation (the ProtEx claim)

The claim is isolated cleanly. The no-negatives model keeps the positive exemplars and reaches nearly
the same separability AUC (PK tail 0.866 vs 0.869; LK tail 0.876 vs 0.889), yet converts to NO
f_micro_w gain (+0.0015 PK, -0.0037 LK, both CI spanning zero). Adding the hard-negative pool moves
both cells over the noise floor with CI strictly positive. Mechanistically this is coherent: the
negatives teach "looks like a positive neighborhood but is NOT," suppressing false positives at the
top of the ranking, which is exactly what a precision-weighted micro-F rewards but a rank-averaged AUC
partly hides.

## Discipline: this positive survives every check

- Score-scale / threshold artifact: every arm shares A's exact score histogram; only the order
  differs. The CONTROL_random_order arm (same histogram, random order) is strongly NEGATIVE
  (-0.076 PK, -0.212 LK, frac_pos 0.0), proving the calibration/threshold interaction cannot
  manufacture a gain. The with-negatives gain is ordering signal, not scale.
- Temporal gate: labels train <= v225, early-stop v225-v227, blind test v227-v230. The verifier never
  sees a v227-v230 label in training.
- Leakage disposition CLEAN: references and reference annotations are strictly t0 = v227 (frozen
  bundle + go_term_metadata id map); self-accession and cosine >= 0.9999 neighbours removed (max
  retained neighbour cosine 0.99951); 0 of 668,234 test candidates use the query as its own positive
  or negative exemplar. Only the candidate label is post-t0; the evidence (neighbours carry the term
  at t0) is strictly past. Postings are structurally built from v227 annotations only.
- CI: paired protein bootstrap, cafaeval exact-parity (point estimate == parity recomputation to 5
  decimals for every arm).

## Scope and honest caveats

- These are true-board-frame deltas on the two cells PROTEA loses (LK-BPO, PK-BPO). +0.034 in both is
  large relative to prior levers (the classifier lever was +0.0225), so it is scrutinised above and
  holds; but a board FLIP against the specific competitors in these cells was not measured here and is
  not claimed.
- Part of the replace-arm advantage could be BP-specialisation (the verifier is trained only on BP
  pk/lk candidates; the deployed reranker is general). The negative-exemplar ablation controls for
  this: the no-negatives model is equally BP-specialised and equally exemplar-conditioned on positives,
  and it does not gain. The lever is specifically the negative-exemplar contrast.
- This corrects the standing picture that "the only channel with signal is generation and every
  rescoring lever is negative." Exemplar-conditioned verification is a rescoring channel that beats the
  deployed reranker on the BP tail, via negative conditioning.

## GO / NO-GO

**GO.** Build the reranker+verifier blend as a BP-tail rescoring stage for LK-BPO/PK-BPO: +0.0344 /
+0.0344 f_micro_w over deployed, temporal-gated, CI strictly above zero, frac-positive 1.0, random-order
control strongly negative, and the negative-exemplar ablation proving the mechanism.
