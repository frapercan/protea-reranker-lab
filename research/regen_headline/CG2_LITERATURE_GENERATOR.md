# CG.2 -- Does a protein's OWN pre-t0 literature GENERATE the true BP-tail the pool misses, at a precision that clears the fill-tax bar?

**Verdict: NO-GO.** A text->GO generator built from each protein's own pre-t0 abstracts
(S-PubMedBert cosine of abstract vs GO-definition, proposing top-k BP terms NOT already in the pool)
adds true BP-tail IA-mass, but at **1.5-1.9% IA-precision** (top5). That is barely above CG.1's
co-occurrence floor (1.0-1.6%) and roughly **6-8x below the 11.59% classifier bar** that itself did
not convert in the true frame. Literature-based generation is NOT meaningfully more separable than
the in-house co-occurrence generator. Per the plan, the gate is not cleared and Step 2 (union +
cafaeval) is not reached.

This is the GENERATION question measured as **SET MEMBERSHIP** (added true vs added false IA-mass),
never a score, so it dodges the scale confound that capped the refuted text-as-scorer test
(`project_text_as_generator_scale_confounded_2026_07_18`). Same reframe, different conclusion made
sharper: the literature channel does reach the tail (30-32% of the missed IA-mass recoverable at
top50) but cannot **separate** it from the false tail, so it re-confirms the SEPARABILITY wall from
outside the sequence-KNN + co-occurrence closure.

Frozen data only. No live DB, no dispatch, no network (all abstracts cached). No cafaeval needed for
the gate (pure IA-mass accounting over pool / ground truth / literature generator).

---

## Abstract coverage (pre-t0, leakage-audited)

PMIDs re-derived here from the frozen UniProt cache, restricted to `publicationDate` year < 2025
(strictly before the 2025-09-04 t0 cutoff; conservative). Refs dated >= 2025 were dropped.

| cell | targets | with pre-t0 PMID + cached abstract | coverage | >=2025 refs dropped |
|------|:---:|:---:|:---:|:---:|
| LK-BPO | 523  | 514  | **98.3%** | 27  |
| PK-BPO | 4,402 | 4,338 | **98.5%** | 254 |

Coverage is near-total: literature is available for essentially every target, so the ~1.5% precision
is a property of the channel, not of missing data.

## Pool reachability (generator-independent ceiling; reproduces CG.1 exactly)

| cell | gt true IA-mass | pool captures | FN IA-mass (pool misses) | FN as % of gt |
|------|:---:|:---:|:---:|:---:|
| LK-BPO | 9,207.1  | 73.9% | 2,398.9  | 26.1% |
| PK-BPO | 80,039.2 | 71.4% | 22,923.9 | 28.6% |

Identical to CG.1's ceiling (same obo/IA/pool/gt), confirming the accounting is apples-to-apples.

## Generation precision (the decider): true vs false IA-mass ADDED, per proposals-per-protein

| cell | k | true IA added | false IA added | **IA precision** | FN recovered |
|------|:--:|:---:|:---:|:---:|:---:|
| LK-BPO | 5  | 368.6   | 19,190    | **1.88%** | 15.4% |
| LK-BPO | 10 | 471.1   | 35,046    | 1.33% | 19.6% |
| LK-BPO | 25 | 584.1   | 77,782    | 0.75% | 24.4% |
| LK-BPO | 50 | 714.8   | 144,366   | 0.49% | 29.8% |
| PK-BPO | 5  | 2,183.8 | 144,358   | **1.49%** | 9.5% |
| PK-BPO | 10 | 2,853.3 | 269,467   | 1.05% | 12.5% |
| PK-BPO | 25 | 4,058.6 | 614,356   | 0.66% | 17.7% |
| PK-BPO | 50 | 5,417.6 | 1,135,109 | 0.48% | 23.6% |

Precision decreases monotonically with k (top5 > top50), so S-PubMedBert *is* ranking above random,
the retrieval signal is real (consistent with the earlier AUC 0.78 / 5.6x retrieval finding). But the
absolute added-true precision is ~1.5-1.9%: recovering even 15-30% of the missed tail requires
injecting **50-200x more false IA-mass**. Under `prop=fill` / global-tau micro-F, no threshold banks
a ~1.5% true fraction.

## The bar, side by side (top5 IA-precision)

| generator | LK-BPO | PK-BPO |
|-----------|:---:|:---:|
| co-occurrence (CG.1, in-house) | 1.61% | 1.03% |
| **literature (CG.2, this)** | **1.88%** | **1.49%** |
| classifier generator (bar) | 11.59% (did not convert) | 11.59% (did not convert) |

Literature is ~1.2-1.4x the co-occurrence floor and ~6-8x **below** the classifier bar. The retrieval
signal that looks strong as AUC (a protein's abstracts point at its own functions above random)
collapses to ~1.5% once measured as added-true SET-MEMBERSHIP precision against the full BP definition
universe, because the false tail retrieved at the same cosine is 50-200x larger by IA-mass.

---

## GO/NO-GO

**NO-GO** on a per-protein literature text->GO generator for LK-BPO and PK-BPO. Added-true IA-precision
is 1.5-1.9% (top5), not meaningfully above CG.1's ~1% co-occurrence floor and far below the classifier
generator's 11.59% (which itself did not convert in the true frame). The missing BP tail is *reachable*
from literature (30-32% at top50) but not *separable* from the false tail. Step 2 is not reached by
plan. This sharpens the campaign's SEPARABILITY wall: even an external, protein-specific literature
channel cannot separate the true missed tail, so the wall is not a coverage/closure limit of the
in-house signals.

## Receipts (under `storage/regen_headline/`)
- `cg2_step1_gate.py` + `cg2_step1_gate.json` -- the gate: coverage + pool reachability + literature generation precision, per cell, per k.
- `cg2_step1_gate.log` -- run log.
- `CG2_LITERATURE_GENERATOR.md` -- this file.
- `cg2_verdict.json` -- machine-readable verdict.
