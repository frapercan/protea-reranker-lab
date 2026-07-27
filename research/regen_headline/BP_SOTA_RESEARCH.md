# BP GO-prediction: SOTA gap analysis and the not-yet-tried levers (deep-research, 2026-07-21)

Cited deep-research synthesis (25 claims verified 3-0, 0 refuted). Full run:
task w3bw3bo6r / workflow wf_4eefcd4f-1d0. Targets our diagnosed wall (protein->term
BP assignment separability; ~40-45% unreachable from t0; PK-BPO novel-tail-on-rich-proteins).

## The gap analysis independently CONFIRMS our diagnosis

- **PK-BPO is the field's recognized OPEN frontier, not our failure.** The official 2026 CAFA5
  assessment introduced the partial-knowledge setting (~70% of newly accumulated annotations, will
  dominate future evals); top methods perform WORSE there because no one optimized for it; framed as
  "an important new challenge and opportunity for methodological innovation."
  (biorxiv 10.64898/2026.04.27.716980)
- **The BP separability wall is a recognized STRUCTURAL fact.** Similar chemistry / molecular function
  maps to a broad range of biological processes (one-to-many, context-dependent), so any sequence/MF
  signal underdetermines BP. Historic BP Fmax capped 0.00-0.31 vs MF 0.01-0.75. This is a within-
  candidate SEPARABILITY wall, not a recall limit -- exactly our diagnosis. (PMC3584852)
- **SOTA winners do NOT solve it.** ProtBoost (CAFA5 2nd) = CondProbMod over the GO tree + GCN stacking;
  PROTGOAT (4th) = pure 5-PLM ensemble + TF-IDF literature + taxonomy -> dense NN; InterLabelGO+ = ZLPR
  rank loss x IA-weighted F1. All improve assignment via GO-hierarchy consistency + multi-PLM ensembling
  -- mechanisms/modalities we already ruled out. STAR-GO, DeepGO-SE do not beat TransFew on BP.
- **TransFew's BP edge is METRIC-DEPENDENT and narrow.** On Test_all BP it leads only on IA-weighted
  Fmax (0.4067); it ranks 2nd on plain Fmax (0.4489 < NetGO 3.0's 0.4716) and far lower on AUPR (0.244
  vs 0.410). Its win is confined to the exact IA-weighted-micro-F family our board uses -> a calibration
  / rare-term-weighting effect (frequency-partitioned MLP experts + IA weighting), NOT a separability
  breakthrough. Its modalities (BioBERT GO-text + GO-DAG GCN autoencoder + ESM2-t48 cross-attention) are
  exactly our ruled-out set. (PMC11374024)
- **Data modalities:** STRING network gives its LARGEST BP lift (~14% AUPR over sequence-only) but uses
  channels we tested/ruled out; biomedical text is structurally WEAK for BP (text-KNN 17% vs 62% MF,
  below the 28% sequence baseline); 3D structure is credited by CAFA5 but is in-family for PLMs.

## The 3 genuinely NOT-YET-TRIED levers (ranked by impact x feasibility for us)

1. **ProtEx-style exemplar-conditioned VERIFICATION** (bioRxiv 2024.05.30.596539). Reframe assignment as
   yes/no verification: for each (protein, term), a retriever fetches BOTH positive AND negative
   class-conditioned exemplars and a Transformer conditions its decision on them. Targets our EXACT wall
   (true-vs-false separability on the reachable tail), a paradigm distinct from plain kNN transfer.
   **No new data; compute-feasible; directly on-target. TOP pick.**
2. **Cross-species PHYLOGENETIC PROFILING** (Pellegrini PNAS 1999; PLoS Comput Biol 0030237; domain-copy-
   number variant ~90% precision top bin). Presence/absence co-occurrence across genomes links
   functionally related proteins; orthogonal to sequence/network/text; **inherently leakage-safe (genome
   content is stable, not annotation-derived); cheap on 12GB.** Targets the unreachable mass.
3. **PERTURBATION TRANSCRIPTOMICS / genome-scale Perturb-seq** (Replogle et al. Cell 2022). Direct
   experimental BP proxy: clusters genes into pathways by correlation of single-cell transcriptional
   responses; assigns BP to sequence-unreachable genes (e.g. C7orf26->Integrator). **Highest ceiling for
   the ~40-45% unreachable mass**, BUT: coverage limited to genes with a detectable phenotype in assayed
   human cell lines (K562/RPE1); needs strict t0-gating for leakage-safety; outputs clusters not GO
   probabilities (needs a mapping step). Biggest, hardest, lowest-coverage.

## Bonus cheap board-adjacent test (frozen data, no new modality)

Is TransFew's BP edge its frequency-partitioned RARE-TERM EXPERTS + IA-weighted CALIBRATION rather than
its (ruled-out) label modalities? Our own findings say the metric rewards calibration. **Graft the
rare-term-expert / IA-calibration mechanism onto our reranker with NO new data** and test under the
temporal gate. Cheapest possible experiment; directly tests the "TransFew wins on calibration not
modality" hypothesis.

## Open questions the research flagged
- Does DeepGO-SE (BP Fmax leader over TransFew) use a materially different neuro-symbolic / semantic-
  entailment mechanism that could help PK-BPO novel terms? (not in our ruled-out set; worth checking)
- Can a t0-gated Perturb-seq / LINCS-L1000 feature map to GO-BP and add within-protein separability on
  the reachable tail for human proteins in the atlases? What is the test-set coverage overlap?
- Under our temporal IA-weighted-micro-F gate, does ProtEx positive+negative exemplar verification beat
  plain kNN on the LK-BPO/PK-BPO reachable tail?

## Caveats
Leaderboard scores are self-reported preprint metrics (not our temporal cafaeval gate); magnitudes do
not transfer. Perturb-seq is demonstrated in 1-2 human cell lines, outputs clusters not probabilities,
and coverage is bounded. ProtBoost's electronic-annotation (GOA/IEA) gain is a known near-leakage source
under strict temporal splits and is NOT recommended. Several PDFs were 403 on direct fetch, verified via
cached/search text.
