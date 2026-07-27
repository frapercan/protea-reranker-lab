# ProtST reranker A/B result (sealed 227->230 frame, 2026-07-13)

Offline A/B, same harness / same pool, the ONLY delta = the EXCLUDE set (arm A excludes the 3
protst_text columns = champion baseline; arm B includes them = champion + protst). Enrichment via the
producer's kNN over the canonical ProtST bank `bd3cd470` (in-memory numpy, 100% coverage).

## 9-cell f_micro_w

| cell    | armA (base) | armB (+protst) | delta      |
|---------|-------------|----------------|------------|
| nk-mfo  | 0.6367      | 0.6249         | -0.0118    |
| nk-bpo  | 0.4607      | 0.4911         | **+0.0304** |
| nk-cco  | 0.5488      | 0.5380         | -0.0108    |
| lk-mfo  | 0.5570      | 0.5483         | -0.0087    |
| lk-bpo  | 0.4901      | 0.5093         | **+0.0192** |
| lk-cco  | 0.5054      | 0.5364         | +0.0310    |
| pk-mfo  | 0.4707      | 0.4704         | -0.0003    |
| pk-bpo  | 0.3433      | 0.3525         | **+0.0092** |
| pk-cco  | 0.4886      | 0.5093         | +0.0207    |
| MEAN9   | 0.5001      | 0.5089         | **+0.0088** |

## Rigour (full A/B agent, 2026-07-13)

- **Harness validated: arm A REPRODUCES the sealed champion** - arm A mean per-cell 0.5001 vs sealed
  0.5006, 7/9 cells exact (nk-mfo/nk-bpo/nk-cco/pk-mfo/pk-bpo/pk-cco identical), lk within 1-iteration
  LightGBM noise. So the arm-B deltas are DECISION-GRADE (same pool, frame, seed; EXCLUDE-only diff).
  The sealed pool was recovered from MinIO (s3://protea/datasets/clean-learned-train227-test230/, sha
  775611822dd9, embedding d8979601, K=30, 64 features), enriched with the EXACT deployed producer
  apply_protst_text (verified byte-identical to _knn_vote on samples).
- **Bootstrap CIs (protein-level paired) - all 3 BP cells EXCLUDE ZERO:**
  - NK-BPO (leakage-free anchor) +0.0284, 95% CI [0.0059, 0.0569], 100% of resamples positive.
  - LK-BPO +0.0197, 95% CI [0.0014, 0.0357], 97.5% positive.
  - PK-BPO +0.0083, 95% CI [0.0017, 0.0153], 98.3% positive.
  The leakage-free NK-BPO anchor moves the SAME direction and by the MOST -> the lift is the ProtST
  TEXT signal, not leakage.
- **Length-stratified BP lift holds all bands:** short(<=300) +0.0056, mid(301-600) +0.0173, long(>600)
  +0.0113.
- **Coverage non-degenerate:** protst_text_score finite on ~60% of BP candidate rows (eval 60.5%, train
  59.7%), full quantile spread; all 7575 eval queries covered.

## Verdict

- **ProtST DENTS the BP wall through the reranker.** All three BP cells lift, including the two
  previously-unwon wall cells: LK-BPO +0.0192, PK-BPO +0.0092, plus NK-BPO +0.0304. CCO also gains
  (lk-cco +0.031, pk-cco +0.021). Directionally consistent with the leakage-free kNN prior
  (+0.072 lk-BPO, +0.037 pk-BPO at kNN; attenuated but positive through the reranker).
- **Net positive: mean9 +0.0088.**
- **NOT a clean 9/9 as run:** adding protst to ALL category models costs ~0.01 on three MF/CC cells
  (nk-mfo -0.012, nk-cco -0.011, lk-mfo -0.009; pk-mfo ~flat). protst is a BP/text-aligned signal;
  fed globally it slightly dilutes the MF/CC models. Single-seed A/B, so part of the ~0.01 wobble may
  be training noise, but the direction (BP up, some MF/CC down) is coherent.
- **"reach #1" is NOT decided by this A/B** - it measures the delta vs OUR champion, not the CAFA
  leaderboard gap on the 2 wall cells. The lift (+0.019 LK-BPO, +0.009 PK-BPO) may or may not clear the
  leaderboard #1 on those cells; that needs the platform eval + leaderboard comparison.

## Multi-seed (seeds 42 + 7) - the MF/CC "regressions" are mostly noise (AB_MULTISEED.md)

Two-seed directional check (LightGBM's single `seed` cascades to all RNG knobs, verified). Delta =
armB - armA per cell:

| cell   | s42 delta | s7 delta | sign      |
|--------|-----------|----------|-----------|
| nk-mfo | -0.0118   | +0.0108  | FLIP (noise) |
| nk-bpo | +0.0304   | +0.0121  | both +    |
| nk-cco | -0.0108   | +0.0017  | FLIP (noise) |
| lk-mfo | -0.0087   | -0.0175  | both -  (real, small) |
| lk-bpo | +0.0192   | +0.0335  | both +    |
| lk-cco | +0.0310   | +0.0412  | both +    |
| pk-mfo | -0.0003   | +0.0007  | FLIP (~flat) |
| pk-bpo | +0.0092   | +0.0059  | both +    |
| pk-cco | +0.0207   | +0.0180  | both +    |
| mean9  | +0.0088   | +0.0118  | both +    |

- **BP wall lift is sign-stable across both seeds** (all 3 BP cells positive both times).
- **CCO lift is sign-stable** (lk-cco, pk-cco positive both times).
- **The MF/CC "regressions" are MOSTLY SEED NOISE:** nk-mfo and nk-cco flip sign across seeds; pk-mfo is
  ~flat. The ONLY consistent loser is **lk-mfo** (-0.009 / -0.018, small).
- **mean9 net positive in both seeds** (+0.0088 / +0.0118).

Upshot: global protst injection helps BP everywhere + CCO everywhere with only ONE small consistent MF
cost (lk-mfo). So global injection is nearly a clean win; a BP+CCO-gated injection would protect lk-mfo
while keeping the sign-stable BP/CCO lift.

## Representation gate (2026-07-13, protst_repr/REPR_RESULT.md) - k-WTA does NOT help protst

Author flagged the untested asymmetry (champion = learned k-WTA over Ankh-base; protst used RAW). Screened
raw vs z-score vs k-WTA on the CHAMPION knn_confirm harness (reproduces 0.2150 exact, 100% protst coverage
of query/ref/pool). mean9 f_micro_w:
- protst_zscore 0.2485 (+0.0030 vs raw, but FLAT on BP)
- protst_raw    0.2455 (ref; = the text_scorer 0.2455 confirmed on THIS harness)
- protst_kwta_d4096_k128 0.2411 (-0.0043 vs raw)
- protst_kwta_d2048_k128 0.2380 (-0.0074 vs raw)
- champion_L48   0.2150 (ref)
DECISION: k-WTA LOSES on protst (incl BP: nk-bpo -0.007, pk-bpo -0.009) - protst's text-aligned space is
already compact+discriminative, the sparse head discards signal (worst on CCO). z-score not a BP lever.
=> feed RAW protst to the producer; NO representation change; gate PASS. BIG SIDE-FINDING: RAW protst BEATS
the champion as a RETRIEVAL space by +0.0335 mean9, winning ALL 9 cells (BP nk +0.040/lk +0.017/pk +0.029,
CCO biggest). That reframes protst from a BP-only feature to a candidate primary/secondary RETRIEVAL arm -
a strategic question for the author (separate from this feature-promotion; kNN mean9 is NOT the reranked
0.4063, so a better retrieval space does not automatically mean a better final number - it needs its own run).

## The clean promotion (next)

Gate protst so it helps BP without touching the 7 already-won cells: add protst_text only to the
BP-aspect combiner path (or per-(category x aspect) models / a per-aspect feature mask), re-run the A/B
to confirm the BP lift holds and the MF/CC regressions vanish, then the AUTHORITATIVE platform
`run_cafa_evaluation` (evaluation_set 6e41eb5b) for the sealed 9-cell + leaderboard rank. Requires
checking whether train_rerank_227230 is per-category (NK/LK/PK, aspects pooled) or per-(cat x aspect):
if per-category, per-aspect gating needs a feature mask or split models.

## Platform-export follow-up (needed for on-platform reproducibility + the authoritative eval)

The offline A/B is decision-grade, but to reproduce this ON the platform (and to run the authoritative
run_cafa_evaluation which reads the platform-exported dataset), `compute_protst` must be plumbed into the
`/jobs`-facing `ExportResearchDatasetPayload` - it is wired into ExportParityFlags/TrainRerankerAutoPayload
but NOT into ExportResearchDatasetPayload (missing on 03714ec/develop/main). A ~2-line PROTEA fix:
add `compute_protst: bool = False` to ExportResearchDatasetPayload + `"compute_protst": p.compute_protst`
in `_dump_to_stage`'s auto_payload (export_research_dataset.py). Until then the platform export cannot
emit protst columns.

## Honest framing for the thesis

ProtST is now a first-class signal in the method (backend + API config + feature family + producer,
all deployed). It contributes ORTHOGONAL BP-wall signal through the reranker (+0.019 LK-BPO,
+0.009 PK-BPO, mean9 +0.0088). The productization question (global vs BP-gated) is an engineering
choice about where to inject it, not about whether the signal is real - it is.
