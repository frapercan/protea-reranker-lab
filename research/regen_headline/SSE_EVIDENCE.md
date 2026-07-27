# Evidence-aware Sparse Semantic-Entailment (SSE) + the BP wall characterised BY evidence

Frozen v225 evidence (`goa_uniprot_all.gaf.225.gz`), leakage-safe (pre-cutoff = legitimate INPUT; test-side v227->v230 evidence NEVER used as input). Corpus = `generator_frames raw_esm2_3b` (ESM2-3B mean, 2560-d); fixed vocab = base SSE meta terms; K=1; eval proteins held out. Compute `repositories/PROTEA/.venv`.

## Evidence tiers

EXP {EXP,IDA,IPI,IMP,IGI,IEP,HTP,HDA,HMP,HGI,HEP} / PHY {IBA,IBD,IKR,IRD} / COMP {ISS,ISO,ISA,ISM,IGC,RCA} / AUTH {TAS,NAS,IC} / ELEC {IEA}. de-circularisation arms are SUBSETS of the exact base positive set: **noiea** = A-positive with any non-electronic tier support; **exp** = A-positive with experimental (EXP) tier support.

## (a) De-circularised training corpus -- geometry + separability

| aspect | A-positives | GAF-supported | noiea (drop IEA) | exp-only | full valAUC | noiea valAUC | exp valAUC |
|---|--:|--:|--:|--:|--:|--:|--:|
| mfo | 450647 | 0.9904 | 0.9821 | 0.9266 | 0.9793 | 0.9794 | 0.9787 |
| bpo | 1744028 | 0.9826 | 0.9774 | 0.9179 | 0.937 | 0.9351 | 0.9237 |
| cco | 739129 | 0.9872 | 0.9785 | 0.8857 | 0.9748 | 0.9744 | 0.9721 |

### Ranking metric (per-protein separability AUC over deployed pool candidates; SSE vs deployed reranker)

| cell | arm | SSE per-prot AUC | reranker AUC | SSE pooled AUC | n_pos |
|---|---|--:|--:|--:|--:|
| pk-bpo | full | 0.6105 | 0.4923 | 0.6132 | 13339 |
| pk-bpo | noiea | 0.604 | 0.4923 | 0.6011 | 13339 |
| pk-bpo | exp | 0.603 | 0.4923 | 0.5743 | 13339 |
| pk-mfo | full | 0.5843 | 0.6559 | 0.6152 | 2060 |
| pk-mfo | noiea | 0.5808 | 0.6559 | 0.6095 | 2060 |
| pk-mfo | exp | 0.5763 | 0.6559 | 0.6048 | 2060 |
| pk-cco | full | 0.5855 | 0.7099 | 0.6398 | 4101 |
| pk-cco | noiea | 0.5904 | 0.7099 | 0.6464 | 4101 |
| pk-cco | exp | 0.5742 | 0.7099 | 0.6317 | 4101 |
| lk-bpo | full | 0.8377 | 0.8289 | 0.809 | 5954 |
| lk-bpo | noiea | 0.8339 | 0.8289 | 0.7952 | 5954 |
| lk-bpo | exp | 0.8114 | 0.8289 | 0.7387 | 5954 |
| lk-mfo | full | 0.8586 | 0.7949 | 0.8391 | 1080 |
| lk-mfo | noiea | 0.8708 | 0.7949 | 0.844 | 1080 |
| lk-mfo | exp | 0.8655 | 0.7949 | 0.8365 | 1080 |
| lk-cco | full | 0.9241 | 0.8206 | 0.9136 | 2371 |
| lk-cco | noiea | 0.923 | 0.8206 | 0.9107 | 2371 |
| lk-cco | exp | 0.9114 | 0.8206 | 0.9017 | 2371 |

### Calibrated wall check -- SSE as pool-reranker f_micro_w (true frame, PK -known) vs deployed anchor

| cell | arm | SSE pool-reranker f | deployed anchor |
|---|---|--:|--:|
| pk-bpo | full | 0.08374 | 0.14351 |
| pk-bpo | noiea | 0.07951 | 0.14351 |
| pk-bpo | exp | 0.06981 | 0.14351 |
| lk-bpo | full | 0.26629 | 0.31323 |
| lk-bpo | noiea | 0.2583 | 0.31323 |
| lk-bpo | exp | 0.18764 | 0.31323 |

Deployed anchor reproduced as pool-reranker (deployed scores over the same pool): {'pk-bpo': 0.11821, 'lk-bpo': 0.29744}

## (b) Per-protein t0 characterisation-quality calibration prior (on full-arm SSE)

Prior = v225 KNOWN evidence summary per protein [exp_frac, nonelec_frac, log-count, log-breadth, has_exp]. Per-protein AUC is invariant to a per-protein prior, so this is measured where a prior CAN move the metric: cross-protein POOLED AUC + calibrated f_micro_w.

| cell | pooled AUC SSE | +prior | delta AUC | f SSE-cal | f +prior | delta f |
|---|--:|--:|--:|--:|--:|--:|
| pk-bpo | 0.6129 | 0.661 | 0.048 | 0.07741 | 0.06257 | -0.01484 |
| lk-bpo | 0.8087 | 0.8048 | -0.0039 | 0.26545 | 0.25927 | -0.00618 |

## (c) Evidence-typed containment (term v225 experimental-fraction scaling of coverage, full arm)

| cell | per-prot AUC raw | evtyped | delta | pooled raw | evtyped | pool-reranker f |
|---|--:|--:|--:|--:|--:|--:|
| pk-bpo | 0.6105 | 0.4913 | -0.1192 | 0.6132 | 0.4859 | 0.08676 |
| lk-bpo | 0.8377 | 0.4515 | -0.3862 | 0.809 | 0.4626 | 0.22359 |

## The BP wall characterised BY evidence (analysis-only)

**Data caveat:** the v227->v230 annotating evidence of each true term is NOT on disk (only the v225 GAF). Term-level (1) + per-protein (3) are v225 corpus proxies; donor (2) is the KNN neighbour's pre-cutoff evidence (provenance-confirmed, NOT the annotation's own). eval.parquet `evidence_code` = donor, not label.

### PK-BPO  (targets 4344, with-unreachable 2961)

- IA mass: true 48352.4, missing 23266.3, unreachable 21728.0
- **unreachable** term-class (IA-frac): {'absent_v225': 0.235, 'mixed': 0.2421, 'exp_dominant': 0.4287, 'iea_dominant': 0.0941}
- **missing** term-class (IA-frac): {'absent_v225': 0.231, 'mixed': 0.2403, 'exp_dominant': 0.4251, 'iea_dominant': 0.1036}
- unreachable IA-weighted mean exp-frac 0.5144 / elec-frac 0.2359
- missing IA-weighted mean exp-frac 0.5137 / elec-frac 0.241
- donor evidence of REACHABLE true terms (IA-frac): {'EMPTY_nonKNN': 0.8649, 'EXP': 0.0437, 'AUTH': 0.013, 'COMP': 0.0288, 'PHY': 0.0129, 'ELEC': 0.0367}
- per-protein t0 exp-frac: unreachable-carrying 0.4646 vs fully-reachable 0.3748

### LK-BPO  (targets 494, with-unreachable 229)

- IA mass: true 8494.5, missing 2169.7, unreachable 2169.7
- **unreachable** term-class (IA-frac): {'mixed': 0.3053, 'exp_dominant': 0.3379, 'iea_dominant': 0.1384, 'absent_v225': 0.2184}
- **missing** term-class (IA-frac): {'mixed': 0.3053, 'exp_dominant': 0.3379, 'iea_dominant': 0.1384, 'absent_v225': 0.2184}
- unreachable IA-weighted mean exp-frac 0.4636 / elec-frac 0.2693
- missing IA-weighted mean exp-frac 0.4636 / elec-frac 0.2693
- donor evidence of REACHABLE true terms (IA-frac): {'EMPTY_nonKNN': 0.9561, 'EXP': 0.0109, 'COMP': 0.0101, 'AUTH': 0.0065, 'ELEC': 0.0108, 'PHY': 0.0055}
- per-protein t0 exp-frac: unreachable-carrying 0.2896 vs fully-reachable 0.212

## Verdict

**Can encoding evidence help? No encoding converts on the metric; only the per-protein calibration prior (b) moves any cross-protein signal, and it does not convert.** (a) De-circularisation is a NO-GO and a null result at the geometry level: at the propagated-label level only ~2% of BP positives are IEA-only and ~8% are non-experimental, so dropping IEA (noiea) or keeping only experimental support (exp) leaves val-AUC essentially unchanged (bpo 0.937/0.935/0.924) and only REMOVES separability (pk-bpo pooled AUC 0.613->0.601->0.574) and calibrated f_micro_w (0.084->0.080->0.070). The learned entailment geometry is NOT materially driven by circular IEA co-annotation, and de-circularising it only deletes signal. (b) The per-protein t0 characterisation-quality prior is the ONLY evidence encoding that moves a cross-protein metric: pk-bpo pooled AUC 0.613->0.661 (+0.048). But it does NOT convert -- calibrated f_micro_w falls 0.077->0.063 (-0.015). This is the crux the campaign keeps hitting: a +0.048 cross-protein-ranking gain that the IA-weighted micro-F does not reward, because the metric rewards cross-protein CALIBRATION under accept-all flooding, not ranking. (c) Evidence-typed containment (scaling coverage by term v225 experimental-fraction) collapses separability to chance (pk-bpo per-prot AUC 0.611->0.491; lk-bpo 0.838->0.452): experimentally-typed terms are not the high-recall ones, so the scaling deletes the signal. All three arms leave SSE far below the deployed anchor on f_micro_w (pk-bpo SSE 0.084 vs anchor-pool 0.118 / full anchor 0.144).

**What the unreachable tail's evidence composition says about the frontier:** the ~45% of PK-BPO true IA mass that no t0 signal reaches is EXPERIMENTAL-type biology, not a pipeline-reachable IEA/ISS slice. By v225 term-profile the unreachable mass is 43% experimental-dominant terms + 24% mixed + 23% absent-in-v225, and only ~9% IEA-dominant (IA-weighted mean experimental-fraction 0.51 vs electronic 0.24); LK-BPO is the same direction (exp-frac 0.46). Decisively, the proteins carrying unreachable mass are BETTER experimentally characterised at t0 than the fully-reachable ones (t0 exp-frac 0.465 vs 0.375 PK; 0.29 vs 0.21 LK), so this is NOT a poorly-annotated-protein artefact. The frontier is genuinely new experimental annotation on already well-studied proteins, with no electronic/homology footprint at t0 -- which is exactly why encoding, de-circularising, or reweighting the t0 evidence cannot break the wall. The small IEA-dominant slice (~9%) is the only 'pipeline-reachable' remainder and is far too small to move the cell. (Caveat: v227->v230 annotating evidence is not on disk; this uses v225 corpus proxies and provenance-confirmed donor evidence, stated above.)
