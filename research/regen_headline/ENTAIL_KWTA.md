# Clean semantic entailment over our learned k-WTA representation: does the ONE mechanism that showed signal (DeepGO-SE's entailment score) SEPARATE and CONVERT when stripped of the broken machinery?

**Headline.** Clean semantic entailment over our learned k-WTA rep SEPARATES as well as DeepGO-SE (PK-BPO AUC 0.6263 vs reranker 0.491; = DeepGO-SE's 0.626) but does NOT CONVERT (every true-frame f_micro_w delta vs the deployed anchor is negative, rescore + generator + controls, CI excludes 0 on the negative side). DeepGO-SE's random-level was the submission construction / candidate-generation ceiling, not the entailment principle. VERDICT = SEPARATES-BUT-CAPPED: the BP wall is FUNDAMENTAL (separability at the reachable tail does not buy f_micro_w at the metric's operating point), not the broken machinery.

## The model (minimal, fully controlled)

- **Architecture**: DeepGOModel MLP tower (2048 k-WTA d8979601 -> 2560 embed) + ELEmbedding ball space (center c_t + radius r_t per GO term), 5 independently-seeded models, SE score = ensemble avg of sigmoid(x . (c_t+hasFunc) + r_t)
- **Axioms**: t0 2025-07-22 GO EL normal forms nf1(70618) nf2(10177) nf3(10173) nf4(17699), 8 relations incl part_of/regulates/has_part; 9655 trainable BP terms + 30251 axiom-only
- **Stripped**: no GAT/DGL (True), no MF-preds pipeline (True), no deepgo2 harness (True)
- **Temporal gate**: labels v227=t0(2025-09-04) propagated BP; blind eval v227->v230; eval-window-new BP terms absent from training by construction
- **Corpus/training**: 554,378 curated IEA+EXP proteins; 5 seeds x 15 epochs; valid AUC 0.998

The protein tower is a small MLP mapping the learned k-WTA code (d8979601, 2048-d) to a point in the GO-ball space; each GO term is a geometric region (center + radius) trained on the EL normal forms; the entailment score is the ensemble-consensus degree the protein's point is subsumed inside the term's ball. This IS DeepGO-SE's semantic-entailment principle, and nothing else.

## (a) SEPARABILITY -- per-protein AUC on the reachable pool tail (diagnostic)

| cell | SE AUC | reranker AUC | SE - reranker | vs DeepGO-SE 0.626 | n_proteins |
|---|---|---|---|---|---|
| LK-BPO | **0.8667** | 0.8294 | +0.03730 | +0.24070 | 515 |
| PK-BPO | **0.6263** | 0.491 | +0.13530 | +0.00030 | 2885 |

PK-BPO: clean entailment over k-WTA reaches **0.6263**, matching DeepGO-SE's 0.626 and beating the deployed reranker's 0.491 (chance). The separability signal is REAL and reproduced by the clean model. LK-BPO: a small edge (0.867 vs 0.829).

## (b) CONVERSION -- true-frame f_micro_w vs the DEPLOYED anchor (DECIDES)

Frame: prop=fill norm=cafa no_orphans toi, PK exclude=groundtruth_PK_known, cafa_eval under PROTEA/.venv, temporal gate v227->v230. Rescore = volume-matched rank-match of the reranker score multiset in SE order (isolates ranking). Generator = union top-k SE proposals into the pool. CIs are paired protein bootstrap (1000x, exact IA-weighted micro-F).

| cell | anchor (reproduced) | RESCORE delta [CI] p>0 | rand-order ctrl | GENERATOR top5 delta [CI] | prior sweep best-K | matched-vol ctrl |
|---|---|---|---|---|---|---|
| LK-BPO | 0.31323 (0.31323) | -0.04418  p>0=n/a | n/a | n/a | -0.13900 (top5) | -0.15546 |
| PK-BPO | 0.14351 (0.14351) | -0.03520 [-0.04246, -0.02823] p>0=0.0 | -0.09832 | -0.01940 [-0.02386, -0.01502] | -0.06403 (top5) | -0.08710 |

(RESCORE and GENERATOR top5 with CI are this run's independent measurements via the exact IA-weighted micro-F decomposition + paired bootstrap; the prior-sweep best-K column is the corroborating true-frame generator sweep from deepgose_kwta/measure_avg.json, negative at every K. LK rescore delta is the LEAKAGE-CLEAN value; see below.)

Every LEAKAGE-CLEAN conversion delta is NEGATIVE, in both modes and under both controls, with the PK CIs excluding zero (rescore [-0.042,-0.028], generator [-0.024,-0.015], p>0=0). The SE ordering does beat random-order rescore (PK -0.035 vs -0.098) -- the 0.626 separability is real -- but it still loses to the deployed reranker at the metric's operating point. The reachable tail is a tiny IA fraction at catastrophic precision (top25 added-true IA-precision ~0.002-0.004, i.e. 100-300x more false IA than true), so a 0.626 AUC sits far below the anchor's threshold.

### LK-BPO was a SUSPECT: +0.0207 was memorization, not conversion

The raw LK rescore looked POSITIVE (+0.0207). LK, unlike PK, has no `-known` exclusion, yet 100% of LK eval proteins are in the SE training corpus and 47% of LK ground-truth BP terms were already v227 training labels. So SE was reordering the pool to surface terms it had MEMORIZED for these exact proteins, and LK let them count. Building the LK analog of PK's `-known` (each LK protein's v227 t0 BP labels, propagated, excluded at scoring) collapses it:

| LK-BPO frame | anchor | SE rescore | delta |
|---|---|---|---|
| raw (no exclusion, leakage-exposed) | 0.31323 | 0.33393 | **+0.0207** |
| leakage-clean (v227-known excluded) | 0.21094 | 0.16676 | **-0.04418** |

Under the leakage-clean frame LK behaves exactly like PK (-0.044 vs PK -0.035). The one positive this whole experiment produced was a temporal-gate hole in the LK cell, and it closes.

## (c) DeepGO-SE random-level DIAGNOSIS -- model or submission construction?

- DeepGO-SE (faithful, standalone full-vocab, global tau) PK-BPO f_micro_w = **0.01124** (delta -0.13227 vs anchor, CI [-0.14062, -0.12362]).
- Proper-construction ceiling (per-protein topK sweep 3..2000): BP tops at ~0.0126; min-combo top500 0.0078; rankpct 0.0032. Ceiling 0.011.
- Candidate recall ceiling (fraction of true NEW BP terms in DeepGO-SE top-K): {'top50': 0.0, 'top100': 0.001, 'top500': 0.057}.

**Diagnosis.** The 0.011 random-level was the SUBMISSION CONSTRUCTION exposing a candidate-generation ceiling, NOT the entailment principle. On a RESTRICTED reachable candidate set the SE score separates true from false (PK-BPO AUC 0.6263 > reranker 0.491). But as a STANDALONE full-vocab submission the true NEW BP terms are largely OUTSIDE the model's high-scoring set: candidate recall ceiling is 5.7% at top500. Every construction we tried (per-protein topK 3..2000, min/avg ensemble combos, rankpct, floor 0.01) caps BP at 0.011-0.0126 -- barely above the global-tau full-vocab 0.011 -- because you cannot rank into the top what the model never surfaces. So: construction is bad AND the underlying full-vocab candidate set is thin; the 0.626 separability is real but lives on a reachable set the standalone generator cannot build.

## VERDICT

- **LK-BPO**: separability SE-reranker +0.03730, rescore delta -0.04418, generator best delta -0.13900 -> **SEPARATES-BUT-CAPPED (real separability, does NOT convert to f_micro_w -> the wall is fundamental)**
- **PK-BPO**: separability SE-reranker +0.13530, rescore delta -0.03520, generator best delta -0.06403 -> **SEPARATES-BUT-CAPPED (real separability, does NOT convert to f_micro_w -> the wall is fundamental)**

**One line.** Clean semantic entailment over our learned representation SEPARATES as well as DeepGO-SE (0.626) but does NOT CONVERT to f_micro_w -> **SEPARATES-BUT-CAPPED**; and DeepGO-SE's random-level failure was the **submission construction / candidate-generation ceiling, not the machinery and not the entailment principle** -- the BP wall is FUNDAMENTAL (a separability/precision limit at the reachable tail), which is why isolating the clean principle does not move it.
