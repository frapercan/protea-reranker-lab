# Phase 0: pre-t0 abstracts carry a real but modest, diffuse signal for what we miss

`storage/cooc_experiment/phase0_abstracts.py` (+ `.json`, `abstract_cache/`)

## The question

Before building a per-protein abstract pipeline to close the two lost board cells (LK-BP -0.072,
PK-BP -0.076, the prior-knowledge cells where the podium wins on text): do pre-t0 abstracts NAME the
novel BP branches we miss, leakage-audited?

## Method

Sample 120 PK-BP + 60 LK-BP blind proteins. Fetch UniProt-curated references, keep only those
published before t0 (2025-09), fetch those PubMed abstracts. Measure whether the abstract text names
the protein's novel-missed BP terms. Leakage guard (a): novel = gt minus the t0 (v227) annotation
state (read-only DB), so only functions not yet annotated at t0 are tested.

## Result

- **Coverage 180/180**: every sampled PK/LK-BP protein has pre-t0 curated literature. The signal is
  not sparse; it is present exactly where we lose.
- **Exact GO-name match 2.3%** (the floor): abstracts almost never use the GO term name verbatim.
- **Loose match (a specific long biology word of the term present) 51%**, consistent PK (51%) / LK
  (51%).

The loose 51% is uninterpretable without a null, because biology abstracts share vocabulary. The null
(match a matched-count set of RANDOM BP terms against the same abstracts):

| | loose hit rate |
|---|---|
| REAL (protein's own gt-BP terms) | 55.1% |
| NULL (random BP terms) | 35.8% |
| **enrichment** | **1.54x** |

## Reading

The signal is REAL (1.54x over random, not noise) with full coverage, but MODEST and DIFFUSE at the
word level: the 36% null base rate is high because function vocabulary is shared across biology
abstracts. A crude keyword feature would drown in that base rate, which is exactly how `protst_text`
landed at +0.0016 as a reranker feature. The board proves text headroom exists (the podium extracts
it), so the question is not whether the signal is there but whether we can lift it above the diffuse
base rate, leakage-free.

## Two unknowns that decide the build

1. **The encoder.** 1.54x is a word-level FLOOR. A semantic encoder that captures what an abstract is
   about (not which words it contains) could lift it well above 1.5x, or fail to and replicate
   protst_text. Untested.
2. **Leakage guard (b), open.** UniProt references are curated SOURCES; some of the 1.54x may be the
   annotation's own source paper (read pre-t0, annotated post-t0), which is leakage, not prediction.
   Needs the target-window GAF's per-annotation source PMID to close.

## Recommendation: Phase 0.5 (cheap, decisive) before the multi-week build

Do NOT commit the full pipeline on a 1.54x word-level signal. Run two cheap decisive probes first,
both on the already-cached abstracts:
- **Semantic enrichment**: embed each abstract and each GO term's DEFINITION with a biomedical
  sentence encoder (PubMedBERT/SciBERT), measure the enrichment of the protein's own terms vs random
  at the embedding level. If it is much higher than 1.54x (say >=3x), the signal is extractable and
  the build is justified. If it stays ~1.5x, the signal is too diffuse and we stop.
- **Leakage guard (b)**: for the novel terms, check how many are annotated post-t0 with a source PMID
  that IS one of the pre-t0 abstracts we used. That fraction is the leakage ceiling.
Only if both clear does Phase 1 (signal-store pipeline + retrieval-level encoder) begin.
