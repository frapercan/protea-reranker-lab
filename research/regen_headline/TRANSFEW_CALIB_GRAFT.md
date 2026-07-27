# TransFew calibration graft: does the BP edge transfer as a calibration mechanism? (2026-07-21)

**VERDICT: NO-GO.** TransFew's IA-weighted-micro-F BP edge is NOT capturable as a
frequency-partitioned IA-calibration mechanism grafted onto our own deployed reranker scores.
The frequency partition is significantly NEGATIVE on both cells under the temporal gate
(LK-BPO -0.01519, PK-BPO -0.00221; both bootstrap CIs exclude 0). Plain IA-calibration is a
statistical null (as monotonicity predicts). Frozen data, CPU, no new modality.

## Hypothesis under test (from `BP_SOTA_RESEARCH.md`)

Deep-research found TransFew leads BP only on IA-weighted Fmax (0.4067), not on plain Fmax or AUPR,
and credited that narrow edge to a t0-stable STRUCTURAL choice: (a) FREQUENCY-PARTITIONED experts
(rare terms get their own operating point instead of being dominated by frequent-term score mass)
plus (b) IA-WEIGHTED calibration matched to the metric. If the edge is this calibration mechanism
rather than TransFew's (already ruled-out) label modalities, we should capture it with NO new data by
grafting the mechanism at the calibration layer over our reranker's scores.

## Build (one variable: calibration/partition only; candidate set and reranker model UNCHANGED)

Deployed per-category reranker: `repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank/`
(`model_{lk,pk}.txt`, `predictions/{lk,pk}/{lk,pk}.tsv`, `train.parquet`, `eval.parquet`).

Three arms, scored identically in the TRUE board frame:
- **deployed** - raw reranker score (LK-BPO already per-protein min-maxed, as deployed), single global tau.
- **plainIA** - ONE isotonic `score -> P(label)`, IA-weighted, fit on v<=225. A monotone transform, so
  under the global-tau max-F metric it MUST reproduce deployed up to grid discretisation. This is the
  CONTROL that isolates "calibration alone" from "the partition".
- **freqpart** - SEPARATE isotonic per t0-frequency bin (rare/med/common). Non-monotone ACROSS bins, so
  it re-orders rare-term vs frequent-term pairs under the single global tau. This is the TransFew
  frequency-partition mechanism. Only this arm can move the number.

Frequency bins: `reference_annotations.parquet` v227, propagated per-BP-term protein count (t0 corpus
prevalence; internal term-id mapped via `go_term_metadata.parquet`), terciles of log1p(freq) over
candidate BP terms. Isotonic (sklearn `IsotonicRegression`, IA sample-weight, clipped [0,1]) is the
calibration layer; the reranker booster and its candidate pool are untouched.

The bins show the exact structure the mechanism targets - a monotone frequency -> positive-rate gradient:

| cell | bin | fit rows | base rate (P(label)) | cand terms |
|------|-----|---------:|---------------------:|-----------:|
| LK-BPO | rare   | 122,321 | 0.0013 | 5,332 |
| LK-BPO | med    | 149,815 | 0.0073 | 5,191 |
| LK-BPO | common | 270,598 | 0.1656 | 5,237 |
| PK-BPO | rare   | 472,324 | 0.0010 | 6,038 |
| PK-BPO | med    | 2,139,881 | 0.0026 | 6,000 |
| PK-BPO | common | 6,808,434 | 0.0071 | 5,977 |

## TEMPORAL GATE (held) + leakage disposition

- **Temporal gate.** Bins + isotonic fit strictly on `snapshot_pair != v225-v227` (v<=225), applied
  BLIND to the v227-v230 deployed predictions. No in-window ceiling is reported; every number below is
  the deployable blind delta.
- **Leakage - clean.** Frequency bins come from t0 corpus prevalence (structural term property, not the
  label). Isotonic sees only `(score, label)` on the past window. IA weights are the fixed lafa_t0
  `IA.tsv`. No blind-window information enters the fit.
- **Not re-finding the deployed tau.** The plainIA control (a single monotone calibration) reproduces
  deployed to within grid noise (LK -0.00223, PK -0.00062), confirming the harness manufactures no
  gain. The freqpart change is therefore a genuine cross-bin reordering - and it is negative.

## Result (TRUE board frame: prop=fill norm=cafa no_orphans toi; PK -known; f_micro_w BP, max over tau)

Paired bootstrap over proteins, 2000 resamples. Deployed anchor reproduced by our own harness.

| cell | arm | f_micro_w | delta vs deployed | bootstrap 95% CI | P(delta>0) |
|------|-----|----------:|------------------:|:----------------:|-----------:|
| **LK-BPO** | deployed | 0.31323 | - | - | - |
| | plainIA (control) | 0.31099 | **-0.00223** | [-0.00786, +0.00323] | 0.21 |
| | **freqpart** | 0.29804 | **-0.01519** | **[-0.02610, -0.00546]** | 0.001 |
| **PK-BPO** | deployed | 0.14351 | - | - | - |
| | plainIA (control) | 0.14289 | **-0.00062** | [-0.00168, +0.00031] | 0.10 |
| | **freqpart** | 0.14130 | **-0.00221** | **[-0.00420, -0.00018]** | 0.014 |

(Anchor note: our harness reproduces deployed LK/PK-BPO at 0.31323 / 0.14351 vs the stored
`result_9cell.json` 0.31096 / 0.14024 - a ~0.002-0.003 offset attributable to a cafaeval kernel/version
difference. All arm deltas are computed within this one harness against this one anchor, so the
comparison is internally exact; the bootstrap engine matches cafa_eval's point number to <1e-4.)

## Is it the FREQUENCY PARTITION specifically, or just IA-calibration?

It is the partition, and it HURTS. plainIA (single-bin IA-calibration) is a statistical null in both
cells - its CI straddles 0 (LK) or hugs 0 (PK), exactly as monotonicity predicts (a global monotone
recalibration cannot change the reachable threshold sets under a tau-swept max-F). The ONLY arm that
moves the metric is the frequency PARTITION, and it moves it DOWN, significantly, in both cells.

**Mechanism of the loss.** Giving the rare bin its own (raised) operating point promotes low-base-rate
rare-term candidates above the global tau. But those candidates are overwhelmingly false (rarest-bin
P(label) = 0.0010-0.0013), and under `prop=fill` every false rare-term prediction also forfeits its
ancestors' free inheritance. So the partition buys a little recall at a steeper IA-weighted precision
cost - net negative. The deployed reranker, which already ingests `go_term_frequency` as a feature,
orders the frequency/precision tradeoff better than an explicit post-hoc partition can.

## Verdict

**NO-GO.** TransFew's BP edge does not transfer to our system as a calibration mechanism on our own
scores. The result is consistent with the campaign's converged finding that the deployed reranker is
near-optimal and generation is the only channel with signal: recalibration/partition is inert-to-harmful
here, matching the ruled-out objective/architecture/representation/head/rescoring channels. If TransFew's
0.4067 is real it lives in its candidate GENERATION (multi-PLM ensemble + GO-DAG modalities we cannot
add cheaply), not in a rare-term calibration trick portable onto our ranking.

## Receipts

- `storage/regen_headline/TRANSFEW_CALIB_GRAFT.md` (this file)
- `storage/transfew_calib/score_arms.json` - deltas + bootstrap CIs (authoritative)
- `storage/transfew_calib/calib_build.json` - bins, base rates, temporal/leakage config
- `storage/transfew_calib/anchor_check.json` - deployed-anchor reproduction + bootstrap-engine self-check
- code: `storage/transfew_calib/{frame,build_calib,score_arms,anchor_check}.py`
- arm predictions: `storage/transfew_calib/pred_{lk,pk}_{deployed,plainIA,freqpart}/`
