# Does a MULTI-PLM candidate pool convert its extra recall into deployable f_micro_w? NO-GO.

Read-only, frozen data, true frame (lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds `-known`).
The one actionable lever the recall-ceiling diagnostic surfaced, taken through the mandatory temporal
gate. Receipts: `multiplm_pool.py`, `MULTIPLM_POOL_FMICROW.json`, `multiplm_pool.run.log`.

## Verdict

**NO-GO on both cells.** A multi-PLM candidate pool (union of five diverse PLM-family kNN BP
retrievals) banks a large IN-WINDOW positive that **fully reverses under the temporal gate**. Even the
cleanest, most-agreed slice floods f_micro_w at the deployed operating threshold.

| cell | deployed (deployable) | multi-PLM selective (deployable) | **DELTA vs deployed** | boot CI95 | frac boot > 0 | vs matched-vol uniform |
|------|----------------------:|---------------------------------:|----------------------:|-----------|--------------:|-----------------------:|
| LK-BPO | 0.28051 | 0.25370 | **-0.02681** | [-0.0415, -0.0124] | 0.05% | **+0.09461** |
| PK-BPO | 0.16255 | 0.10114 | **-0.06141** | [-0.0739, -0.0494] | 0.0% | **+0.04423** |

Both deployable deltas are negative with bootstrap CIs entirely below zero. cafaeval-parser parity is
**exact** (A_cafa==A_mine, Sel_cafa==Sel_mine to 5 decimals) so the metric is the board's, not a
manual approximation. The FIT-window policy was **not** allowed to self-veto and did not: it chose to
augment (`fit_optimal_is_no_augmentation=false`, phi=0.005 for both) and lost on the forward window.

## The reversal, quantified (this is the whole story)

| cell | FIT in-window delta (phi=0.005) | FIT top-slice extra prec | APPLY (blind) extra prec | APPLY deployable delta |
|------|--------------------------------:|-------------------------:|-------------------------:|-----------------------:|
| LK-BPO | **+0.04498** | 0.371 | 0.062 | **-0.02681** |
| PK-BPO | **+0.19589** | 0.388 | 0.054 | **-0.06141** |

The FIT window "found" gains of +0.045 (LK) and +0.196 (PK) at phi=0.005. That gain is an artefact of
co-fitting the GBM policy, phi, and tau on the same labels the extras are scored against: the policy's
top-0.5% slice looks 37-39% precise **in-window**, but forward-blind on the disjoint APPLY window the
identical policy's top slice is only **5-6% precise**, and at that precision the 94-95% false mass sinks
f_micro_w below the un-augmented baseline. This is the exact in-window-positive-that-reverses failure
the discipline warned about, now the fourth+ retrieval channel to hit it.

## The multi-PLM SELECTION signal is real and generalizes; it just cannot clear the fill-tax

This is the nuance worth banking, and it distinguishes this result from a null:

- **Agreement is a genuine, monotone, temporally-stable precision selector.** Cross-PLM agreement
  (1..5 diverse families proposing a term) stratifies precision monotonically, and the stratum
  precision table is **near-identical FIT vs APPLY** (corr 0.93 LK / 0.99 PK): agree=5 candidates run
  ~24x the precision of agree=1 in both windows. Unlike the classifier lever (whose gain was
  window-specific term memorization), this signal is **not** window-memorization; it forward-transfers.
- **Selection crushes volume.** The multi-PLM selective pool beats a matched-volume RANDOM-uniform
  control by **+0.095 (LK) / +0.044 (PK)** with CIs entirely positive. The selected extras average
  agreement 4.5-4.6 (almost all agree=5) at 5.4-6.2% precision; random extras sit at 0.4-0.6%. So the
  ranking is doing real work.
- **But 5-6% precision still floods.** Even the best-available slice, added at the deployed operating
  tau, contributes ~19x more false IA-mass than true. The wall is **separability, not retrieval or
  ranking**: the terms are reachable (the ceiling's 61%/55% union recall of the missing tail) and the
  best t0 selector puts them at 5-6% precision, but the deployed cell already operates far above that,
  so any admission of this tail is net-negative under prop=fill/global-tau. Confirms the ceiling's
  "reachable-but-inseparable" reading with a live f_micro_w measurement.

## Leakage disposition (clean)

Proposals are t0 sequence-homolog transfers (`c905dffa` = GOA v227 = 2025-09-04 cutoff, self-accession
neighbour dropped, learned champion excluded); STRING is v12.0 (2023) experimental+coexpression only, a
single binary GBM feature. The only post-t0 information anywhere is the candidate LABEL (the window
ground truth). The temporal gate makes even label-side window memorization impossible: FIT labels are
Sep->Nov, APPLY labels Nov->Mar, temporally disjoint.

## Added-candidate precision (APPLY, blind, the honest number)

- LK-BPO: selected extras 6.16% precise (mean agreement 4.54), random-uniform extras 0.37%.
- PK-BPO: selected extras 5.41% precise (mean agreement 4.58), random-uniform extras 0.60%.

## One-line GO/NO-GO

**NO-GO.** The multi-PLM diversity recall channel is real and its cross-PLM-agreement selector is
strong and temporally stable, but even its cleanest slice (agree=5, ~5-6% forward precision) floods
f_micro_w under cafaeval prop=fill: LK -0.0268, PK -0.0614 deployable, both CIs strictly negative. This
completes the retrieval-frontier characterization: every t0 generation channel (classifier extras,
text/abstract, annotation-RAG, STRING, and now multi-PLM homolog diversity) reaches the missing BP tail
but none separates it from the false tail at the metric's operating point. The wall is SEPARABILITY.
