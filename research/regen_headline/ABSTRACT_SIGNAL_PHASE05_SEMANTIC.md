# Phase 0.5: a biomedical semantic encoder lifts the abstract signal from 1.54x to 5.6x. Gate CLEARS.

`storage/cooc_experiment/phase05_semantic_abstracts.py` (+ `.json`, `.log`)

## The question, and the gate

Phase 0's word-match gave 1.54x enrichment: real but diffuse (36% null base rate, biology vocabulary
is shared). A crude keyword feature drowns in that base rate, which is how `protst_text` landed at
+0.0016. The gate before committing the build: can a SEMANTIC encoder tell "the abstract is ABOUT
this function" from "the abstract happens to use this word", lifting the signal well above the word
floor (>=3x)?

## One variable vs Phase 0

Same sample (SEED=42, 120 PK + 60 LK), same novelty guard (a) (novel = gt minus v227 state), same
cached pre-t0 abstracts. Only the matcher changes: string match -> cosine between a biomedical
sentence encoder's embedding of the abstract and of each BP term's DEFINITION. Metric = a per-protein
RETRIEVAL task: positives = the protein's novel-missed BP terms, negatives = all other BP terms
(30,804 with definitions), each scored by MAX cosine over the protein's papers.

Encoder: `pritamdeka/S-PubMedBert-MS-MARCO` (cached, biomedical retrieval, no download).

## The result: CLEARS decisively

| cell | mean AUC | median AUC | recall@top-5% | enrichment over random |
|---|---|---|---|---|
| overall | 0.777 | 0.838 | 0.281 | **5.61x** |
| PK-BP | 0.763 | 0.831 | 0.247 | 4.94x |
| LK-BP | 0.808 | 0.847 | 0.358 | **7.16x** |

Phase-0 word floor was 1.54x. The semantic encoder lifts it to **5.6x** (7.16x on LK-BP, one of the
two lost board cells). Median AUC 0.84 means for a typical protein the encoder ranks its true
novel-missed BP terms well above random: the abstract semantically points at the function we miss.

## Reading, disciplined

This is RETRIEVAL QUALITY (can the encoder rank true terms high among all BP), NOT a board delta.
Two things stay open and must not be overclaimed:

1. **The bridge from AUC 0.78 to a board flip is unproven.** protst_text also had signal and landed
   +0.0016 as a scalar reranker feature. The difference here, and the reason to proceed: the abstract
   encoder is a CANDIDATE GENERATOR (top-k BP terms outside the pool closure), feeding the ONE channel
   with proven signal (generation, the classifier-extras lever +0.02245). Not a scalar feature over
   existing pool candidates. The next test is whether these generated candidates PAY in the cafaeval
   frame, the same way the classifier's did.
2. **Leakage guard (b) is open.** The retrieval uses only pre-t0 abstracts (year<2025) to predict
   post-t0 annotations (temporally leakage-free), but some AUC may be the annotation's own source
   paper (read pre-t0, curated post-t0). For an annotation-PREDICTION system that is the valid task,
   not a cheat; but the thesis claim must state it. Fully closing (b) needs the target-window (v230)
   GAF per-annotation source PMID.

## Decision

The gate CLEARS (5.6x >> 3x). Build justified. First real cost: abstracts for the 4,964 eval PK-BP +
LK-BP proteins (fetch running, cached). Then the decisive "does it pay" test: semantic candidates,
same scorer S as the classifier-extras lever, arms A (pool) / B_classifier / B_semantic / B_union,
scored in the cafaeval legacy frame. If B_union > B_classifier the two generators are complementary
and the path to the two lost cells runs through generation from BOTH the classifier and the text.
