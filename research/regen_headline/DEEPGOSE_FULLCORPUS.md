# DeepGO-SE FULL-CORPUS variant vs baseline: does reaching the rare BP tail change the verdict?

Full-corpus variant of the faithful DeepGO-SE (ELEmbedding over t0 2025-07-22 GO EL axioms + ESM2-3B protein MLP + 10-model semantic-entailment ensemble). The ONLY change vs the baseline is the trainable-corpus threshold MIN_COUNT 50 -> 10, promoting the rare/novel BP tail from axiom-only zero-classes into trainable classes. Temporal gate (train t0<=v225, blind eval v227->v230), true-frame f_micro_w vs the DEPLOYED anchor LK 0.31323 / PK 0.14351, reproduced exactly as A_pool_only in the same harness. AUC never decides.

## Corpus actually used

| | baseline | full-corpus variant |
|---|---|---|
| MIN_COUNT | 50 | 10 |
| trainable BP terms | 3,905 | 9,655 |
| axiom-only zero BP classes | 36,001 | 30,251 |
| train / valid proteins | 57,077 / 6,341 | 57,077 / 6,341 |
| PLM embeddings | ESM2-3B 2560-d, 88k exp | ESM2-3B 2560-d, 88k exp |

574k-curated corpus was rejected: only ProtT5 (1024-d) embeddings are materialized for it on disk (`storage/protex/ref_emb_norm_fp16.npy`, 574,627 x 1024 fp16); using it would confound the corpus change with a PLM-family change and violate the same-PLM-family constraint. Training population is identical (63,418 BP-labelled proteins) at both thresholds: propagation gives every experimental protein a frequent BP ancestor, so lowering MIN_COUNT adds rare LEAF terms to the trainable vocabulary (+5,750), not new proteins. That is the lever this variant isolates.

## LK-BPO (anchor 0.31323)

Anchor reproduced as A_pool_only: baseline 0.31323, variant 0.31323 (matches anchor: True). FN unreachable-tail IA mass 2398.9.

### 1. GATE: recall-added of the FN tail + IA-precision (avg ensemble)

| top-k | metric | baseline | variant |
|---|---|---|---|
| top10 | frac FN tail recovered | 0.0424 | 0.0481 |
| top10 | added-true IA-precision | 0.0022 | 0.0018 |
| top25 | frac FN tail recovered | 0.0746 | 0.078 |
| top25 | added-true IA-precision | 0.0022 | 0.0013 |

(bars: co-occ 0.010, network 0.058, classifier 0.116)

### 2. SEPARABILITY: per-protein AUC on the reachable pool tail (AUC is diagnostic, never verdict)

| | baseline | variant |
|---|---|---|
| SE mean AUC | 0.7496 | 0.7169 |
| reranker mean AUC | 0.8294 | 0.8294 |
| SE - reranker | -0.0798 | -0.1125 |

### 3. CONVERT: true-frame f_micro_w delta vs anchor (THIS DECIDES) with matched-volume + random-order controls

| top-k | | baseline | variant |
|---|---|---|---|
| top5 | delta_B (SE proposals) | -0.16706 | -0.11233 |
| top5 | delta matched-volume uniform | -0.15546 | -0.15546 |
| top5 | delta random-order | -0.18226 | -0.14259 |
| top10 | delta_B (SE proposals) | -0.20690 | -0.13491 |
| top10 | delta matched-volume uniform | -0.15546 | -0.15546 |
| top10 | delta random-order | -0.20803 | -0.15993 |
| top25 | delta_B (SE proposals) | -0.21971 | -0.18173 |
| top25 | delta matched-volume uniform | -0.15544 | -0.15544 |
| top25 | delta random-order | -0.23427 | -0.20067 |

## PK-BPO (anchor 0.14351)

Anchor reproduced as A_pool_only: baseline 0.14351, variant 0.14351 (matches anchor: True). FN unreachable-tail IA mass 22923.9.

### 1. GATE: recall-added of the FN tail + IA-precision (avg ensemble)

| top-k | metric | baseline | variant |
|---|---|---|---|
| top10 | frac FN tail recovered | 0.0263 | 0.0242 |
| top10 | added-true IA-precision | 0.0016 | 0.001 |
| top25 | frac FN tail recovered | 0.0532 | 0.0406 |
| top25 | added-true IA-precision | 0.0018 | 0.0008 |

(bars: co-occ 0.010, network 0.058, classifier 0.116)

### 2. SEPARABILITY: per-protein AUC on the reachable pool tail (AUC is diagnostic, never verdict)

| | baseline | variant |
|---|---|---|
| SE mean AUC | 0.6159 | 0.6139 |
| reranker mean AUC | 0.491 | 0.491 |
| SE - reranker | 0.1249 | 0.1229 |

### 3. CONVERT: true-frame f_micro_w delta vs anchor (THIS DECIDES) with matched-volume + random-order controls

| top-k | | baseline | variant |
|---|---|---|---|
| top5 | delta_B (SE proposals) | -0.08255 | -0.09710 |
| top5 | delta matched-volume uniform | -0.08710 | -0.08710 |
| top5 | delta random-order | -0.08255 | -0.09710 |
| top10 | delta_B (SE proposals) | -0.09649 | -0.10784 |
| top10 | delta matched-volume uniform | -0.10318 | -0.10318 |
| top10 | delta random-order | -0.09649 | -0.10784 |
| top25 | delta_B (SE proposals) | -0.10861 | -0.12200 |
| top25 | delta matched-volume uniform | -0.11929 | -0.11929 |
| top25 | delta random-order | -0.10862 | -0.12200 |

## VERDICT

- **LK-BPO**: best convert delta_B = -0.11233 vs anchor 0.31323; tail recovered top25 base->var = [0.0746, 0.078]; separability SE-reranker = -0.1125. **NO-GO / SAME WALL = SEPARABILITY (reaches the tail but cannot separate true from false)**
- **PK-BPO**: best convert delta_B = -0.09710 vs anchor 0.14351; tail recovered top25 base->var = [0.0532, 0.0406]; separability SE-reranker = 0.1229. **NO-GO / SAME WALL = REACHABILITY (promoting tail terms to trainable did not recover more of the residual)**
