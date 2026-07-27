# We do not reproduce the board, and the inputs needed to fix that are not on disk

This retracts a claim I made earlier tonight and replaces it with the test that actually
answers the question. Scripts: `storage/regen_headline/build_submission.py`,
`withhold_on_real_submission.py`.

## The claim I made, and why it was wrong

I reported "PK-BP reproduces the board, -0.0054". That compared **our recipe scored locally**
(0.21269) against **`predictions_protea.tsv` scored by the board** (0.21807). Those are two
different prediction files. Two different arms landing near each other is not a reproduction,
and reading it as one is exactly the reflex the campaign keeps punishing: it agreed with what I
wanted, so I did not ask what it compared.

## The test that does answer it

`predictions_7401_reranked.tsv` is the one file that exists **both locally and on the board**.
The board's own `results_{NK,LK,PK}` carry it:

| cell | board | our re-run of the same file, board's line |
|---|---|---|
| PK-BP | **0.1170** (n=3,756, cov_w 0.853) | **0.20132** (cov_w 0.8587) |
| LK-BP | 0.3480 (n=480) | 0.30376 |
| NK-BP | 0.3090 (n=244) | 0.3021 |

**Same file, same ground truth, and PK-BP differs by 0.084.** Under the recorded flags rather
than the board's line it is 0.20256, so the flags are not it either. `tau` agrees (0.522 vs
0.520) and `cov_w` nearly agrees (0.8512 vs 0.853): the file sits at the same operating point
and the score still disagrees.

## What is left, and it is not available

The board evaluates with `${t0Dir}/release/go-basic.obo` and `${t0Dir}/release/IA.tsv`
(`evaluation.nf:211-212`). **Neither is on disk.** `CAFA_forever/data/` holds `releases/` and
`catalog.json` and no t0 directory at all; the only ontology and IA in the tree are the lab's
`protea-lafa-knn/lafa_t0_Sep_2025/` copies dated 3 June. f_micro_w weights every term by its
information accretion, so a different IA file rescales the entire metric. That is the remaining
variable and we cannot test it.

**Consequence, stated plainly: no lab number in this campaign is in the same frame as the
board.** Deltas measured locally may well transfer, since a reweighting that is common to both
arms of a comparison largely cancels, but absolute local figures are not board figures and must
never be published as though they were. The one place we can check, we are off by 0.084.

## What this does not touch

The prefilter lever is a **delta measured entirely within one frame**, twice: +0.0092 held-out
(unanimous across ten protein folds) on `eval_scores.parquet`, and +0.0145 on
`predictions_7401_reranked.tsv`. Both arms of each comparison used the same obo and the same IA,
so the reweighting cancels. The lever stands. What does not stand is any claim that our local
0.22288 is a board number.

## A correction to my own discipline #12

I wrote "a gt is identified by `n`, not by its filename" and used it to conclude that the
recorded 0.117 came from a different ground truth. **`n` is per-method coverage, not the ground
truth.** The board's own `results_PK` proves it: one directory, one truth, and
`predictions_protea` n=2,670, `predictions_percutgraft` n=3,517, `transfew` n=3,412. The
principle survives, the application was false, and it had led me to dismiss a real discrepancy
as an artifact. **A rule invented to catch a mistake can manufacture one.**

## The board's own table, read without a story

Our four arms, all against the same PK ground truth:

| arm | cov_w | f_micro_w |
|---|---|---|
| `predictions_protea.tsv` | 0.604 | **0.2181** |
| `predictions_percutgraft.tsv` | 0.798 | 0.1400 |
| `predictions_7401_reranked.tsv` | 0.853 | 0.1170 |
| `protea-knn-v1.tsv` | 0.788 | 0.0750 |

Our best arm is also our least covering, by a wide margin, which is what the withholding
mechanism predicts. It is **not** a clean ladder: `protea-knn-v1` covers less than
`7401_reranked` and scores worse, and among external methods `transfew` covers 0.776 and scores
0.2943 while `funbind` covers 0.786 and scores 0.2348. Coverage is not the only axis and this
table is not evidence that it is. It is recorded here as an observation, not as a finding.

---

## Why the frame differs, and why it is not ours to fix

`nextflow.config` names `democafa_package = "/work/idoerg/ahphan/democafa_package"`. **CAFA_forever
is the benchmark host's pipeline and the frozen results under `data/releases/` were computed on
their cluster with their t0.** The t0 was never lost: it was never ours. `params.output_root`
resolves `t0Dir` to `${projectDir}/${timepoint_id}`, and no such directory has ever existed here.

Every ontology on this box, by its own declared version:

| `data-version` | path |
|---|---|
| **releases/2025-07-22** | `protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo`  <- the one the whole lab uses |
| releases/2026-03-25 | `protea-neural-head/data/go-basic.obo` |
| releases/2026-01-23 | `lafa-smoke/data/go-basic.obo` |
| releases/2025-06-01 | `lafa_workdir/data/go-basic.obo` |

**The directory is named `lafa_t0_Sep_2025` and holds a July 2025 ontology.** The board's release
is `Sep_2025_Mar_2026`, so its t0 is September 2025. No September 2025 ontology exists anywhere on
the machine, including the archive partition.

What that does and does not explain, tested rather than assumed: our obo carries 48,165 terms and
**misses zero** of the ground truth's 5,021; our IA carries 39,906 and **misses zero** as well. So
the gap is not absent terms. It is the **IA values**: information accretion is computed from the
annotation corpus at t0, so a July corpus weights the same terms differently from a September one,
and f_micro_w weights every term by exactly that. This is consistent with the 0.084 and is not
proof of it. Without the board's IA the hypothesis cannot be closed, and it is recorded as open.

## The resolution, which is a discipline rather than a blocker

We do not need to reproduce the board's absolutes, because **the board publishes them**. The
frozen `results_{NK,LK,PK}/evaluation_best_f_micro_w.tsv` are authoritative and already scriptable:
`storage/regen_headline/board_nine_cell.py` reads them. So:

- **Any board figure we publish is cited from the board's own table.** Never recomputed locally,
  never "reproduced". 0.3374 / 0.4402 / 0.2181 are their numbers and they are correct as cited.
- **Any lab figure we publish is a delta, within one frame, both arms sharing an obo and an IA.**
  The prefilter lever qualifies: +0.0092 held-out across ten protein folds, +0.0145 on a real
  submission file.
- **The two are never mixed, and a local absolute is never presented as a board absolute.** That
  mixing is exactly what produced the withdrawn "PK-BP reproduces, -0.0054".

Under that rule the campaign is not blocked and nothing published is wrong. What is blocked is the
one thing we would like: predicting, before submitting, what the board will score. Closing that
needs the host's t0, which sits on their cluster. **The author's standing instruction is not to
contact them until the thesis ships, so this is a decision for the author and not a task to route
around.**

The alternative that does not involve them is rebuilding the September 2025 t0 from public
archives: the GO release plus the matching GOA corpus, which is what `timepoint_build` mode is for.
That is a real download and a real rebuild, and it is not a frozen-data operation, so it is not
something to start unasked.


---

# The frame, bounded: four axes tested, all flat. It is not the ontology and not the IA.

Scripts `storage/regen_headline/which_ia_is_the_boards.py` and `which_obo_is_the_boards.py`, data
in the matching `.json`. One variable each, on `predictions_7401_reranked.tsv`, the only file that
exists both here and on the board, which the board scores at PK-BP **0.1170**.

| axis | tested | result |
|---|---|---|
| the board's line vs the lab's | `resolve_contradiction.py` | **-0.00002** |
| bpo-only vs all-aspect rows | `resolve_contradiction.py` | **+0.00000** |
| **the IA table** | 3 candidates | 0.20132 / 0.20376 / 0.20076 |
| **the ontology** | 4 candidates | 0.20132 / 0.20189 / 0.20817 / 0.20426 |
| the ground truth | verified pair-for-pair | identical |

**Every axis is flat and the 0.084 survives all of them.**

## Two claims of mine, withdrawn

**"The directory name lies."** It does not. `protea/core/band_registry.py` pins band **v227** to
GOA release **227 (2025-09-04)** with ontology **`releases/2025-07-22`**, and a CI guard
(`scripts/check_band_registry.py`, wired into `lint.yml`) enforces it. The September GOA release
was built against the July ontology: that pairing is deliberate, documented in
`docs/source/architecture/evaluation.rst`, and tested. `lafa_t0_Sep_2025/` is the LAFA t0 for
September 2025 and holds exactly the obo that t0 pinned. The ontology was never wrong, and the
obo sweep confirms it changes nothing anyway.

**"It is the IA values."** Also withdrawn. `docs/IA_PROVENANCE_v227.md` had already compared the
three IA tables on 2026-06-06 and chosen ours because it is "the exact table the deployed LAFA
endpoint scored its predictions against". The sweep confirms the choice is not the discrepancy:
all three land at 0.201 to 0.204, including the generic CAFA6 table the note explicitly rejects.

That note also names, in its own words, the thing we spent the night rediscovering: mixing a
snapshot or IA across bands "inflates a **phantom PROTEA-vs-LAFA gap**". The mechanism was known.
This measurement shows it is not what we are looking at.

## What is left, and why it stops here

Two suspects survive and neither is resolvable from disk:

1. **The board's `predictions_7401_reranked.tsv` may not be our `predictions_7401_reranked.tsv`.**
   A filename is not an identity. Against this: the optimal threshold agrees **exactly** at 0.520,
   which a different score distribution has little reason to do.
2. **The evaluator's version.** The board activates its own `${params.cafaeval_env}`; we run the
   `cafaeval-protea` fork. A different implementation is a different measurement, and this is the
   one axis we cannot vary because we do not have theirs.

The operating rule is unchanged and does not depend on closing this: **board figures are cited from
the board's frozen table and lab figures are published only as within-frame deltas.** Nothing we
publish rests on the 0.084 being explained.

**#18: when every axis you can vary is flat, the cause is in an axis you cannot vary. Name it and
stop, rather than re-testing the flat ones with more precision.**
