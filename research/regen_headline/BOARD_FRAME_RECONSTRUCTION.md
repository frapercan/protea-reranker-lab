# The board frame is RECONSTRUCTED. The 0.084 gap was a missing flag, not a missing host t0.

Verdict: **RECONSTRUCTED.** The board's evaluation frame is fully reproducible from
on-disk artifacts. All three BP cells of the one file that exists both here and on the
board (`predictions_7401_reranked.tsv`) reproduce the board's published number to within
0.0004. The prior `WE_DO_NOT_REPRODUCE_THE_BOARD` verdict was wrong, and it was wrong for
the exact reason the campaign kept warning about: it read part of the command line and
generalised it.

Scripts (scratchpad, re-runnable): `repro_7401.py`, `repro_pk_known.py`,
`lever_in_true_frame.py`. Board pipeline: `CAFA_forever/modules/local/evaluation.nf`.

## What the board actually runs, read in full this time

The NK and LK evaluations (`evaluation.nf:211-215` and `243-247`) are:

    cafaeval <t0Dir>/release/go-basic.obo <predDir> groundtruth_{NK,LK}.tsv \
      -ia <t0Dir>/release/IA.tsv -out_dir results_{NK,LK}/ \
      -toi groundtruth_terms_of_interest.txt \
      -prop fill -norm cafa -threads N -no_orphans

The **PK** evaluation (`evaluation.nf:263-280`) is the same PLUS one line the other two
do not have:

      -known groundtruth_PK_known.tsv

`-known` maps to `cafa_eval(..., exclude=...)` (`cafaeval-protea/__main__.py:30,60`;
`evaluation.py:777`). It excludes the annotations a protein was already known to hold at
t0, so PK is scored only on newly gained function. **Every prior lab measurement of a PK
cell omitted it.** No `-max_terms`, no `-th_step` anywhere in the pipeline, so the defaults
hold: `max_terms=None` (vectorised parser), `th_step=0.01`.

## The reproduction

`predictions_7401_reranked.tsv` (464,785 rows, scores in [0.0001, 1.0], **zero
non-positives**) is the only file present both on disk and in the board's frozen
`results_{NK,LK,PK}`. Scored with the lab's `lafa_t0_Sep_2025/{go-basic.obo,IA.tsv}` and
the exact board line above:

| cell | board (published) | reconstruction | Δ | cov_w board/ours | tau |
|---|---|---|---|---|---|
| PK-BP | **0.117** | **0.11666** | 0.0003 | 0.853 / 0.8526 | 0.52 / 0.52 |
| LK-BP | **0.348** | **0.34829** | 0.0003 | 0.918 / 0.9178 | 0.49 / 0.49 |
| NK-BP | **0.309** | **0.30897** | 0.00003 | 0.914 / 0.9139 | 0.49 / 0.49 |

pr_micro_w agrees too (PK ours 0.0989 vs board pr_w 0.099). The frame is reconstructed.

## The 0.084 gap, decomposed

The `WE_DO_NOT_REPRODUCE` receipt reported PK-BP off by 0.084 (board 0.117 vs re-run
0.20132) and, after four flat sweeps (IA x3, obo x4, board-line vs lab-line, bpo vs
all-aspect), concluded the cause lay in an axis it could not vary: a September-2025 host
t0 obo/IA absent from disk, or the board's own cafaeval version. **That conclusion is
refuted.** The gap decomposes entirely into two on-disk, testable pieces:

| component | effect on PK-BP | evidence |
|---|---|---|
| **the `-known` exclude flag** (PK only) | **0.20132 -> 0.11666 = -0.0847** | `repro_pk_known.py`; adding `-known` alone closes the whole gap |
| parser path (`max_terms` None vs 500) | **0.00000** | `repro_7401.py`: identical to 5dp both ways, because the file has zero non-positives |
| obo/IA vintage | ~0 | already-flat sweeps; and the reconstruction now lands on the board with the lab's obo/IA |

The entire 0.084 was `-known`. The four sweeps were flat because none of them varied the
one flag that mattered; every arm sat at ~0.201 because every arm omitted `-known`. The
lab's `lafa_t0_Sep_2025` obo/IA ARE functionally the board's for these files: no separate
host t0 is needed.

Note on the parser: the 0.0933 legacy-vs-vectorised divergence
(`PARSER_DIVERGENCE.md`) is **real but scoped to files carrying non-positive scores**
(`eval_scores.parquet`, raw lambdarank margins). Every actual submission file
(`predictions_7401_reranked`, `predictions_protea`, `predictions_percutgraft`) is
probability-scaled with no non-positives, so the parser path is inert for them. The parser
was never the board gap for a real submission.

## Provenance caveat, stated because it bounds the claim

The board rows for the PROTEA team's own files are rounded to exactly 3 decimals
(`protea-knn-v1` 0.075, `predictions_7401_reranked` 0.117, `predictions_percutgraft`
0.140), whereas every host-run method carries full float precision (`transfew`
0.2943164210958724, `predictions_protea` 0.21806691440311407). The `.bak_prereranked`,
`.bak_pregraft`, `.bak_preinterpro` staging files show these PROTEA rows were spliced into
the frozen table in local editing passes, and they are not in the host's method list
(`nextflow.config enabled_window_methods`). So the 0.117/0.348/0.309 we reproduce were
computed **locally by the author** with the lab obo/IA, not by the host pipeline.

What this means, precisely:
- The frame in which OUR methods were scored for the board comparison IS the lab frame,
  and it is fully reconstructable (proven above to <0.0004).
- `transfew` was scored by the host. The comparison OUR-method-vs-TransFew is valid only
  if lab obo/IA equals the host t0 for the shared terms. The release is pinned to one t0
  (band_registry: GOA 227 / obo releases/2025-07-22, CI-guarded), the author deliberately
  built `lafa_t0_Sep_2025` to match, and the reproduction is exact, so the frames coincide
  to within the <0.0004 reproduction noise. A within-frame delta above ~0.001 is a real
  board-frame movement.

## The two lost cells, now measurable in-frame

The lost cells are on the deployed submission `predictions_protea.tsv`:

| cell | PROTEA (protea) | TransFew | gap |
|---|---|---|---|
| LK-BPO | 0.44018 | 0.51202 | **-0.0718** |
| PK-BPO | 0.21807 | 0.29432 | **-0.0762** |

`predictions_protea.tsv` itself is still absent from disk (a genuinely separate,
probability-scaled arm; its board row is full-precision and was host-run with `-known`).
That blocks reconstructing the deployed arm's absolute number, but it no longer blocks
measuring levers in the board frame: the recipe (obo + IA + line + `-known` for PK) is now
known exactly and reproduces the on-disk board arm.

## Objective 3: the levers re-measured in the TRUE frame

Prefilter lever (withhold bottom-fraction by score) on the on-disk board arm
`predictions_7401_reranked.tsv`, in the true frame (`-known` applied to PK):

| withhold | PK-BP f | LK-BP f |
|---|---|---|
| 0% (as built) | 0.11666 | 0.34829 |
| 50% | 0.11911 | 0.35237 |
| 70% | 0.12038 | 0.35485 |
| 90% | **0.12081** | **0.35551** |

- PK-BP max gain **+0.00415** (vs TransFew 0.2943, gap -0.173). **No flip.**
- LK-BP max gain **+0.00722** (vs TransFew 0.5120, gap -0.156). **No flip.**

**The `-known` correction shrinks the PK lever 3.5x.** The `withhold_on_real_submission`
receipt measured +0.0145 on this same file for PK, but in the wrong frame (no `-known`);
in the true frame it is +0.004. The mechanism reproduces (monotone positive, recall rises
as withheld cells return as free ancestors) but the magnitude in the true PK frame is far
too small to close -0.076.

Classifier-extras generation lever (`score_the_extras.py`, +0.02245 at small k): NOT
re-measured here because it operates on `eval_scores.parquet` candidate sets and needs the
generation pipeline re-run, which is a larger job than frozen-data scoring. **But its prior
magnitude is now suspect for the same reason**: it was measured on PK without `-known`, and
the prefilter case shows `-known` roughly thirds a PK gain. It must be re-measured in the
corrected frame before any PK claim rests on it. That is the single highest-value follow-up.

## Verdict

**RECONSTRUCTED.** The board frame is reproducible from on-disk artifacts to <0.0004 on all
three BP cells of the shared file. Exact recipe:

    obo = protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo   (releases/2025-07-22)
    ia  = protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv
    cafaeval OBO predDir groundtruth_{NK,LK,PK}.tsv -ia IA \
      -toi groundtruth_terms_of_interest.txt -prop fill -norm cafa -no_orphans
    PK ONLY: add  -known groundtruth_PK_known.tsv
    defaults: max_terms=None (vectorised), th_step=0.01

**Consequence for the campaign.** The blocker recorded in
`project_cafaeval_parser_divergence_2026_07_17.md` ("no lab number is in the board's frame,
the host t0 is not on disk, contacting the host is the author's decision") is dissolved.
Lab experiments CAN be scored in the board's own frame today, with no host contact and no
rebuild. The one residual gap is the missing `predictions_protea.tsv` (the deployed arm's
own predictions), which blocks reconstructing that specific arm's absolute but not the frame.

**Consequence for the flip.** Neither named lever flips LK-BPO or PK-BPO. In the true frame
the prefilter buys +0.004 (PK) / +0.007 (LK) on the 7401 arm against gaps of -0.17, and the
classifier-extras lever's PK magnitude is very likely overstated by the same missing-`-known`
error and must be re-measured before it can be trusted. The honest state: the frame is now
open for measurement, and no lever measured to date closes either lost cell in it.

## Discipline

The prior verdict spent a night sweeping four axes and named the cause as an axis it could
not vary. The cause was a flag it could vary and did not read: `-known`, on line 279 of the
same file whose lines 211-215 it had quoted. #11 once more, and this time it cost a false
"irreconstructable": read the WHOLE command, including the process you did not open.
