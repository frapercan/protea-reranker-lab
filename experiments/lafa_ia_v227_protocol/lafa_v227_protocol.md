# F-LAFA-IA.1: LAFA protocol pinned at v227, IA verified, minimal-op verdict

Analysis slice. No exports, no MinIO writes, no training. All findings are
reproducible from the scripts in this directory plus the read-only DB
queries recorded in `band_shift.json`. Verified 2026-06-04.

Artifacts:

- `reconcile_ia.py` + `ia_reconciliation.json`: the IA-source reconciliation (item 2).
- `band_shift.json`: the v226 vs v227 band positive-set delta (item 3).
- this file: the authoritative protocol, the verdict, and the recommendations.

## 1. The exact LAFA deployed protocol (pinned to the v227 band)

The deployed LAFA submission scores predictions with `cafaeval` against
the CAFA_forever release windows whose t0 is `Sep_2025` (GOA v227). The
authoritative rule set, cross-checked against
`protea-lafa-knn/score_and_inject_windows.sh` (the deployed scoring
script), `protea-lafa-knn/protea_knn_predict.py` (the prediction step),
and the CAFA_forever release layout:

| Rule | LAFA deployed value | Source |
|------|---------------------|--------|
| t0 (train cutoff) | Sep 2025 = GOA v227 | `reference/manifest.json` `annotation_release: goa@227`; `lafa_t0_Sep_2025/` |
| Eval window | t0 to release-end, e.g. `Sep_2025_Mar_2026` = v227 to about v230 | `CAFA_forever/data/releases/Sep_2025_*` |
| Query pool | 7401 fixed queries | `lafa_queries_7401.fasta` (7401 sequences) |
| Reference pool | 471160 ProtT5 vectors @ goa v227 (no post-t0 leakage) | `reference/manifest.json` `n_reference: 471160` |
| Embedding | ProtT5 `Rostlab/prot_t5_xl_half_uniref50-enc`, mean pool, L2 norm, fp16 | `reference/manifest.json` |
| IA file | `lafa_t0_Sep_2025/IA.tsv` (see item 2 for the authoritative pick) | `score_and_inject_windows.sh` `IA=$T0/IA.tsv` |
| OBO | `lafa_t0_Sep_2025/go-basic.obo`, `data-version: releases/2025-07-22` | OBO header |
| Propagation (`prop`) | `fill` (max-propagate scores to all ancestors) | `score_and_inject_windows.sh` `-prop fill` |
| Normalization (`norm`) | `cafa` | `-norm cafa` |
| Orphan handling | `no_orphans` ON (drop terms with no IA / out of universe) | `-no_orphans` |
| Terms of interest | `-toi groundtruth_terms_of_interest.txt` per window | `score_and_inject_windows.sh` |
| `max_terms` | not capped in the deployed LAFA scoring | absent in `score_and_inject_windows.sh` |
| `th_step` | cafaeval default (0.01) | not overridden |
| `n_cpu` / threads | 4 | `-threads 4` |
| GT split | NK / LK / PK, each scored separately; PK adds `-known groundtruth_PK_known.tsv` | `score_and_inject_windows.sh` |
| Prediction transfer | top_k ProtT5 neighbours, evidence-weighted score, max over hits, then True-Path max-propagation | `protea_knn_predict.py` |
| Prediction format | 3-col TSV `Query_ID  GO_Term  Score`, no header, scores in [0,1] | `predictions_7401.tsv` (295111 rows) |
| GT construction | window-scoped newly-true experimental annotations, split NK/LK/PK by prior-knowledge state at t0 | CAFA_forever `groundtruth_{NK,LK,PK}.tsv` |

This is the ONE authoritative rule set. The crucial point for the rest of
the slice: the headline LAFA metric is the IA-weighted micro Fmax
(`evaluation_best_f_micro_w.tsv` column 31, used in the inject script
validation), NOT the unweighted Fmax. That matches brief section 5:
report the IA-weighted number, not the propagation-inflated unweighted
Fmax.

### How the lab eval sweep differs from the LAFA protocol

The lab sweeps (`phase3d_K*_<plm>_sweep.py`) call `cafa_eval` with:

```
prop="fill", norm="cafa", no_orphans=True, max_terms=500, th_step=0.001, n_cpu=1
```

Differences against the LAFA deployed protocol:

1. NO `ia=` (the lab sweep produces unweighted Fmax only). F-LAFA-IA.0 (PR
   #57) is what adds the `ia=` re-eval. Closing the gap to LAFA means
   passing `ia=datasets/ia/IA-swissprot-exp-v227.txt` (item 2).
2. NO `toi` (terms of interest filter).
3. `max_terms=500` and `th_step=0.001` in the lab vs no cap and default
   `th_step=0.01` in LAFA. Evaluator knobs, not data; they change the
   threshold grid and per-query term cap, not the underlying band.
4. The lab `gt` is candidate-restricted (only positives that appear among
   reranked candidate rows), so the lab Fmax is conditional on retrieval;
   LAFA gt is the full window newly-true set. This is the optimism brief
   section 5 flags.
5. Band: the lab eval band is `v226->v230` (next item); LAFA is `v227->v230`.

## 2. IA verified: the two IA sources reconciled

Both IA tables are the same goa v227 SwissProt experimental(+IC+TAS)
corpus run through democafa IA. They differ ONLY because each was
propagated against a different OBO snapshot. Numbers from
`ia_reconciliation.json`:

| Metric | Value |
|--------|-------|
| lab `IA-swissprot-exp-v227.txt` terms | 38739 |
| knn `lafa_t0_Sep_2025/IA.tsv` terms | 39906 |
| shared terms | 38650 |
| shared with identical value | 28386 |
| shared with diff > 1e-9 | 10264 |
| median abs diff (nonzero) | 0.0070 bits |
| mean abs diff (all shared) | 0.0774 bits |
| roots (BP/CC/MF) both exactly 0.0 | yes |

The 89-vs-1256 term-universe gap and the ~10k small per-term diffs trace
to the OBO snapshots: lab uses `releases/2026-01-23`
(`datasets/bench-v1-K5/go.obo`, md5 `74281b5c...`, 38739 terms after the
democafa cleaning), knn uses `releases/2025-07-22` (`go-basic.obo`, md5
`3e716fc3...`, 48165 raw `[Term]` stanzas). The handful of large diffs
(one term swinging the full 14.6 bits) are terms whose parent set changed
or that became obsolete between the two GO releases.

Authoritative pick: the lab `IA-swissprot-exp-v227.txt`, because its OBO
(`releases/2026-01-23`) is the SAME ontology the lab eval grid propagates
predictions and ground truth against (`datasets/bench-v1-K5/go.obo`).
cafaeval requires the IA table, the prediction propagation, and the GT
propagation to share one OBO; mixing the knn `2025-07-22` IA into a
`2026-01-23` eval can drive IA negative or drop terms from the universe.
For scoring the LAFA reproduction against its own `2025-07-22` windows the
knn `IA.tsv` is the matched file. The two are interchangeable in substance
(median diff 0.007 bits); they are NOT interchangeable across OBO
snapshots. Provenance for the lab file is fully documented in
`datasets/ia/provenance.md` (democafa commit `742814eb`, corpus
`5e84d6c5...`, 547133 raw experimental annotations, 88103 proteins).

## 3. MINIMAL-OP VERDICT: aligning the bench grid to v227 needs a re-export, not a re-score

Verdict: a re-score of the existing v226-cut predictions against a
v227->v230 eval band is NOT a faithful v227 alignment. A full re-export at
a v227 train cutoff is required for a correct LAFA-comparable grid.

Proof on the prostt5 K3 probe cell
(`datasets/bench-v1-K3-v226-lineage-prostt5`):

- The eval.parquet bakes in `label`, `snapshot_pair` (`v226-v230`),
  `distance`, and the full KNN feature set (`vote_count`, `neighbor_*`,
  alignment stats, `evidence_code`, anc2vec, PCA, lineage). The manifest
  pins `eval_snapshot_pair: v226-v230` and 13 train pairs ending
  `v220-v226`. So the candidates and every feature are derived from a
  v226 reference pool, and the positives in `label` are the v226->v230
  newly-true set. None of that is recomputable by changing an evaluator
  flag; it is frozen in the parquet.
- The band actually moves a lot (`band_shift.json`, read-only DB):
  - v226->v230 newly-true: 125026 pairs (the grid positives).
  - v227->v230 newly-true: 79199 pairs (the LAFA band positives).
  - Moving t0 from v226 to v227 removes 45827 positive pairs (~37% of the
    grid positive labels). Those are pairs that became true in the
    Aug->Sep 2025 window; they are `label=1` in the bench parquet but
    must not count as newly-true under the LAFA v227 band.

A "re-score against v227->v230" would at best relabel the existing
v226-pool candidate rows, which (a) leaves the candidates and all features
v226-derived (the KNN neighbours were chosen from a v226 reference that
may include proteins annotated only after v227, i.e. mild post-t0
leakage), and (b) cannot recover any v227-band positive that the v226
retrieval never surfaced as a candidate. So even the cheap path is not a
clean v227 grid; it is a v226 grid with a v227 gt overlay, which mixes two
bands. The honest options:

| Path | What it produces | Disk cost (root, currently 47G free) |
|------|------------------|--------------------------------------|
| A. Re-score only (relabel parquet vs v227 gt) | v226-pool candidates + v227 labels. Mixed band, ~37% of positives shift; mild post-t0 leakage in candidates. NOT LAFA-comparable. | Near zero (rewrites the `label` column of existing parquet, under 1 GB per cell, in place). |
| B. Full re-export at v227 train cutoff | True v227 grid: candidates + features + labels all at v227. LAFA-comparable. | Each cell export writes a new MinIO dataset (train+eval parquet, sqlite align-cache). Per memory the export pipeline grows root with no hot reclaim; 24 cells is the heavy slice. FORBIDDEN here (root 95%). |

Conclusion: the cheap re-score is methodologically wrong for a
LAFA-comparable claim, and the correct path (full re-export) is exactly
the disk-heavy operation this slice is gated against. The 24-cell v227
grid (F-LAFA-IA.1b) must wait for disk headroom (export reclaim / external
partition) and should be a full re-export at the v227 train cutoff, not a
relabel.

Caveat for the thesis: the bench grid stays a valid v226-lineage benchmark
and the F-LAFA-IA.0 IA-weighted delta over KNN baseline is computed
honestly on it. What it is NOT is numerically identical to the deployed
LAFA leaderboard number, because of the one-band (v226 vs v227) and the
candidate-restricted-gt differences. The delta (reranker minus KNN on
wFmax) is the publishable claim; the absolute is band-specific.

## 4. Fate of the hung phase2-lafa-v227 chain: RETIRE

The `phase2-lafa-v227` chain (log `~/Thesis2/phase2-lafa-v227.log`) is
dead. Last entry 2026-06-03T08:13:09Z; no live process. It dispatched one
predict_go_terms job (`7fe1bd80-...`, K=3 ankh_base) which sat at "0 preds"
across three polls, the exact predict_go_terms coord-never-claims
signature (memory `project_predict_coord_stuck_2026_05_30`: coord acks RMQ
but never UPDATEs job.status, RMQ re-delivers every 3 min duplicating
pred_sets).

Recommendation: formally retire it, do not unblock. Rationale: that chain
was a v227 predict fan-out that writes new prediction_sets to the DB and
(downstream) MinIO datasets, i.e. the exact disk-heavy path this slice
forbids while root is at 95%. It is also subsumed by the item-3 verdict:
the correct v227 alignment is a full re-export, not a v227 predict over the
v226 grid. Unblocking it would burn the coord-recovery recipe (purge queue
+ dedupe pred_sets + force-finalize) on a path we have just shown to be the
wrong shape. Retire it; when disk frees up, F-LAFA-IA.1b does the clean full
v227 re-export and the predict fan-out is part of that, with the coord-claim
fix (FIX-PREDICT-COORD-CLAIM) landed first.

## 5. Metrics-doc collision: metrics.rst (PR #56) is canonical; deduplicate PR #57

State on develop today:

- PR #56 `docs(F-LAFA-IA-DOC)` is MERGED. It added
  `docs/source/metrics.rst` (463 lines, comprehensive: IA definition,
  four-metric taxonomy, equivalence table) and it is already wired into
  `docs/source/index.rst` toctree as `metrics`.
- PR #57 (`feat/lafa-ia-baseline-reeval`, F-LAFA-IA.0) is OPEN and adds a
  SECOND metrics page `docs/source/lafa_ia_metrics.rst` plus a toctree
  entry `lafa_ia_metrics`, alongside its experiment code
  (`experiments/lafa_ia/reeval_ia_baseline.py`).

Recommendation (record only, do not merge here): `metrics.rst` is the
canonical single metrics page. Before PR #57 merges, drop
`docs/source/lafa_ia_metrics.rst` and its toctree line from #57, and fold
any genuinely new content (the concrete IA-weighted seed42 baseline numbers
from F-LAFA-IA.0) into `metrics.rst` as a results subsection. That keeps
#57 as a code+results PR and avoids two competing metrics pages in the
toctree (which would also trip the Sphinx build on a duplicate-caption
ambiguity). If #57 already merged by the time this is actioned, a follow-up
doc PR should delete `lafa_ia_metrics.rst`, merge its content into
`metrics.rst`, and remove the stray toctree entry.

## Reproduce

```
# item 2
python experiments/lafa_ia_v227_protocol/reconcile_ia.py \
  datasets/ia/IA-swissprot-exp-v227.txt \
  ~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv \
  experiments/lafa_ia_v227_protocol/ia_reconciliation.json

# item 3 (read-only, needs the PROTEA DB up; queries in band_shift.json)
```
