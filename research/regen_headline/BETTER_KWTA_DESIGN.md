# Regenerate the champion representation "properly" (author 2026-07-11: "regeneramos mejor, justificamos y documentamos")

## The problem with the current champion
`d8979601` = learned k-WTA code (dict=2048, top_k=128, objective=hard-neg) over base `08234f06`
= **Ankh-base, LAST layer (layer 48), mean-pooled, NO z-score** (normalize=f). The layer is selected by
`layer_indices=[0]` which the ankh backend maps to `hidden_states[-(0+1)] = hidden_states[-1]` = the last
layer. It was generated in exploratory/debugging scripts (storage/layer_ablation/*.py), not as a clean
platform artifact. Two things about it are, on our own evidence, suboptimal:
1. It sits on the **last layer (48)**, which our layer ablation flags as suboptimal.
2. It does **not z-score** the base, and z-scoring is the ONE lever the crowning experiment found positive.

## The evidence (storage/layer_ablation/scale_train.log, 100k, 3 seeds, Wilcoxon+Holm)
Clean, WITHIN the local extraction harness (mean9 at KNN level, ~0.14-0.16):
- **L10-std vs L48:  +0.0167  (p_holm 6.78e-33, sig)**  <- layer 10, z-scored, beats last layer
- **L10-std vs L10:  +0.0185  (p_holm 3.16e-39, sig)**  <- z-score IS the lever
- L10     vs L48:  -0.0018  (ns)                        <- layer alone (no std) does nothing
- mix-learned vs L10-std: -0.0009 (ns)                  <- multi-layer mixing does NOT beat single L10-std
=> best evidence-backed representation = **L10-std** (Ankh layer 10, z-scored) + k-WTA(128) hard-neg.
   Single layer; multi-layer/concat not worth the complexity.

## THE CONFOUND (be honest, do NOT oversell the +0.0167)
Same log, line 53: `CONTROL L48 vs served champion: mean9 L48=0.1422  champ=0.2150`.
The SERVED champion (0.2150) massively outperforms the LOCAL L48 extraction (0.1422) in this harness.
So the whole L10/L48/std ablation runs on an extraction path that does NOT reproduce production. The
+0.0167 is a clean signal INSIDE the local harness, but it does NOT prove L10-std beats the PRODUCTION
champion. The only way to know: regenerate L10-std THROUGH THE PRODUCTION pipeline and benchmark it
head-to-head against the faithful champion baseline. The win is a hypothesis, not a result.

## Plan (rigorous, platform-native, justified + documented)
1. **Baseline = the faithful regeneration already running** (job d0bc085e, champion d8979601 as-is) ->
   gives the honest CURRENT number = the control the "better" is measured against. Keep it.
2. **Build the L10-std champion candidate through PRODUCTION** (not debugging scripts): a new base
   EmbeddingConfig = Ankh layer 10 + z-score (fit mean/std on the reference pool, standardize per-dim),
   then train the k-WTA (dict 2048, top_k 128, hard-neg) on it -> a new learned-code EmbeddingConfig.
   This is the real engineering: getting L10 + z-score + k-WTA into a reproducible platform pipeline.
   Secondary candidate to keep in reserve: concat-L10L48-std.
3. **Materialize embeddings** for pool + queries on the new config (compute_embeddings).
4. **Regenerate the pipeline** on the new config (export -> rerankers -> predict -> cafaeval).
5. **Benchmark head-to-head** vs the faithful baseline, STRATIFIED (9 cells, length x category x
   neighbor-identity, CIs on deltas, leakage-free). This resolves the confound.
6. **Decide + document:** if L10-std beats the champion through production -> promote as the new SOTA
   representation, write an ADR + a WRITEUP, and the regenerated headline uses it. If it does NOT ->
   document that the debugging-era L48 champion was actually near-optimal through production (the confound
   explained), and keep it. Either outcome is a rigorous, publishable finding.

## Open engineering questions to resolve before step 2
- How is z-score applied in the platform? Config `normalize` may be L2, not standardization; z-score
  (per-dim mean/std fit on the pool) may only exist in the offline scale scripts -> may need a clean
  platform mechanism (a base-config flag or a fit-and-store std vector).
- How is the k-WTA trained/registered as an EmbeddingConfig via the platform (apply_learned_encoder /
  _learned_code_embed / compute_embeddings)? Reuse the saved heads in storage/learned_encoders/
  (ankh_base_hardneg.pt) as reference, but retrain on L10-std.
- Layer 10 extraction: layer_indices convention is hidden_states[-(li+1)]; layer 10 of 48 => li=38.
  Verify against extract_layers.py in the ablation.
