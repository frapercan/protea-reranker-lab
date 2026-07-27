# CG.1 -- Does a protein's own t0 known annotations GENERATE new true BP-tail terms the pool misses?

**Verdict: NO-GO (in-house).** An in-house generator P(new BP term | known set), built from frozen
t0 co-occurrence, does add *some* true BP-tail IA-mass the deployed pool lacks, but only at
**0.4-1.6% IA precision** (98-99% of the mass it adds is false). That precision cannot survive
cafaeval's `prop=fill` / global-tau micro-F, so the gate is not cleared and Step 2 is not reached (by
plan). The result is consistent with, and sharpens, the campaign's converged "SEPARABILITY wall":
the missing true tail is *reachable* (26-29% of true IA-mass sits outside the pool) but not
*separable* from the false tail by any in-house co-occurrence signal. **CG.3 (external network
channel) is justified.**

This is the GENERATION question, kept distinct from the refuted RESCORING one
(`TRANSITION_SEPARABILITY_PROBE.md`): every candidate scored here is a (protein, BP-term) pair that
is **not** already in the pool for that protein.

Frozen data only. No live DB, no job dispatch, no cafaeval needed for the gate (pure IA-mass
accounting over pool / ground truth / generator).

---

## The gate numbers (per cell)

Co-occurrence source: `reference_annotations.parquet` (frozen **v227 = t0**, cutoff 2025-09-04),
strictly pre-eval-window (gains are v227->v230), the only full annotation corpus on disk. Generator
score s(t | K_p) = sum over seeds k in K_p of PPMI(k, t), seeds -> BP terms. Propagation and IA from
the lab `lafa_t0_Sep_2025` obo/IA. A proposed BP term is TRUE if it is an ancestor-or-self of a
gained ground-truth term.

### Pool reachability (generator-independent ceiling)

| cell | gt true IA-mass | pool captures | FN IA-mass (pool misses) | FN as % of gt |
|------|:---:|:---:|:---:|:---:|
| LK-BPO | 9,207 | 73.9% | 2,399 | **26.1%** |
| PK-BPO | 80,039 | 71.4% | 22,924 | **28.6%** |

The pool already holds ~72-74% of the true IA-mass. A perfect generator could add at most the FN
mass (26-29% of total). **Correction to the "52% of FN mass absent from pool" figure:** at the
IA-mass level the pool misses only 26-29%, not 52%. The 52% claim is closest to PK's gained-*leaf*
FN *count* (40.2%, 16,754/41,727 leaves); by IA-mass the tail the pool misses is materially smaller.

### Generator precision (the decider): true vs false IA-mass ADDED, per proposals-per-protein

| cell | k | true IA-mass added | false IA-mass added | **IA precision** | FN recovered |
|------|:--:|:---:|:---:|:---:|:---:|
| LK-BPO | 5  | 417   | 25,456  | **1.61%** | 17.4% |
| LK-BPO | 10 | 571   | 49,249  | 1.15% | 23.8% |
| LK-BPO | 25 | 804   | 116,563 | 0.69% | 33.5% |
| LK-BPO | 50 | 1,056 | 216,706 | 0.49% | 44.0% |
| PK-BPO | 5  | 933   | 89,445  | **1.03%** | 4.1% |
| PK-BPO | 10 | 1,479 | 184,241 | 0.80% | 6.5% |
| PK-BPO | 25 | 2,816 | 473,872 | 0.59% | 12.3% |
| PK-BPO | 50 | 4,181 | 946,096 | 0.44% | 18.2% |

Precision decreases monotonically with k (top5 > top50), so the co-occurrence generator *is* ranking
above random -- the signal is real -- but the absolute level is ~1%. To recover even 18-44% of the
FN mass you must inject ~100x-200x more false IA-mass. Under `prop=fill` every false term forfeits
its ancestors' free TP inheritance and adds FP mass, so no global threshold can bank the ~1% true
fraction. For comparison, the full-GO classifier generator (`score_the_extras_trueframe`) proposes at
**11.59%** precision -- an order of magnitude better -- and *still* did not convert to a board win.
A ~1% co-occurrence generator has no path to a positive deployable f_micro_w.

**Gate = NOT cleared.** Per the plan, Step 2 (union + cafaeval) is not reached: the generator adds
negligible *separable* true mass. Confirming a negative f_micro_w with cafaeval would only restate
the refuted rescoring result and the classifier-generator cap.

---

## LK leakage disposition: CLEAN (with empirical proof), not blocked

The board freezes a t0 known set only for PK. For LK I obtained the t0 MF/CC known set from
`reference_annotations.parquet` (frozen v227 = t0), a provably pre-t0, pre-eval-window artifact, and
used it only cross-aspect (F/C seeds -> BP proposals). Leakage self-checks:

- 523/523 LK targets carry a frozen F/C known set (mean 9.16 terms).
- **BP-terms-in-any-LK-known-set = 0.** This is decisive: the LK cell is *defined* as "lacks BP at
  t0". If `reference_annotations` carried any post-t0 leakage, LK targets would show their gained BP
  terms -- they show exactly zero. This empirically confirms the reference is the clean t0 state for
  these proteins, and the known set is aspect-disjoint from the BP targets.

So LK-BPO is delivered leakage-clean, distinct from the prior campaign's leaky +0.029 (which used a
post-t0 source). No BLOCKED-on-t0-export needed.

---

## GO/NO-GO

**NO-GO** on an in-house known-annotation co-occurrence generator for LK-BPO and PK-BPO. The true BP
tail the pool misses is reachable (26-29% of true IA-mass) but not separable from the false tail by
any t0 co-occurrence signal (generation precision ~0.4-1.6% IA). This confirms and quantifies the
SEPARABILITY wall and justifies CG.3 (external network / literature channel), where the missing
signal must come from outside the sequence-KNN + intra-organism co-occurrence closure.

## Receipts (all under `storage/regen_headline/`)
- `cg1_step1_gate.py` + `cg1_step1_gate.json` -- the gate: pool reachability + generator precision, per cell, per k.
- `CG1_KNOWN_ANNOTATION_GENERATOR.md` -- this file.
- `cg1_verdict.json` -- machine-readable verdict.
