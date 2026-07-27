# The prior-knowledge channel: located, measured, and mostly artefact (2026-07-17)

Author's question: we lose LK-BPO and PK-BPO, the two cells where the protein already
knows something. Do we use what it knows in the OTHER aspects, or what its neighbours
know? Answer: we do, through one first-order counter, and it is worth almost nothing.
Then: is that channel thin by design or by implementation? Four experiments, three
controls, and the honest answer is "there is something, and two thirds of what we first
measured was an artefact".

## 1. The structural hole, and it is real

The benchmark's own definitions make the question sharp. **LK = the protein has
annotations in SOME aspects at t0 and gains terms in OTHERS. PK = it already had terms
in that same aspect.** So **LK-BPO is literally "has MF/CC, lacks BP"**: the cell where
cross-aspect knowledge is the only prior-knowledge asset there is.

**Verified from data, not from the docs:** of the 103 LK-BPO proteins present in the
benchmark's frozen `groundtruth_PK_known.tsv`, the median count of known BP terms is 0
and **not one of them has a single known BP term**. The definition holds.

What that cell has to work with (LOFO, `lofo_9cell/run.py`; sign convention read from the
source: `delta = baseline - dropped`, **positive means the family helps**):

| family | NK-BPO | LK-BPO | PK-BPO |
|---|---|---|---|
| `classifier` | **+0.198** | **+0.104** | **0.000** |
| `association` (the cross-aspect counter) | 0.000 | **+0.0037** | **+0.0174** |
| `anc2vec_query` (the protein's own known terms) | 0.000 | **-0.0001** | **+0.0161** |
| `anc2vec_neighbor` (the neighbours' terms) | **+0.009** | **-0.0012** | +0.0003 |
| `go_context` | +0.011 | +0.0074 | +0.0018 |
| `lineage` | 0.000 | +0.0008 | **-0.0038** |

Read the LK-BPO column. `anc2vec_query` is **mute by construction**: it scores similarity
against the protein's own known **BP** terms, which that cell does not have. The
neighbours' terms **hurt** (-0.0012): they help when nothing is known (+0.009 on NK) and
stop helping the moment the protein knows something. That leaves **`association`, one
family, +0.0037**, while the gap to TransFew on that cell is **+0.072**.

And `association_cross` is built by `build_go_cooccurrence`: a **first-order count**.
It cannot express that THIS molecular function implies THIS process; it blurs "co-occur
because one implies the other" with "co-occur because both are frequent".

**The regime pattern this explains.** On the board we are **first on NK-BPO (0.3374),
ahead of TransFew (0.3005, fourth)**, and third on LK-BPO (0.4402 vs 0.5120) and PK-BPO
(0.2181 vs 0.2943). We win where nothing is known and lose the moment something is. In
MF and CC we win all three regimes. **This is not a Biological Process wall. It is a
prior-knowledge-exploitation wall that only surfaces in BP**, the branch where knowing
some terms implies others.

## 2. Four experiments, each one shrinking the last

Every arm predicts a protein's NEW BP terms from its t0-known terms ALONE: no sequence,
no embedding, no neighbours. `C` = counting (what we ship). `S` = set-level: Jaccard-kNN
in KNOWN-TERM space, transferring BP from proteins whose known SET resembles this one.

### 2a. Full vocabulary (`cross_aspect_channel_ceiling.py`)

| arm | f_micro_w | AUC |
|---|---|---|
| C counting | 0.2047 | **0.9393** |
| S set-level | **0.4172** | 0.8569 |
| O oracle | 0.9684 | |

`S - C = +0.2125`. **And AUC ordered them backwards**: counting has the better AUC and
half the f_micro_w. That is the fourth time on this campaign that AUC has pointed the
wrong way.

**But the oracle exposes the regime**: 0.9684 at recall 0.9388 means every protein got
the WHOLE vocabulary as candidates (4,475,548 rows / 1,321 proteins = 3,388 = the vocab).
This is a full-vocabulary ranking task, **not the deployed pool** (~140 candidates,
recall 0.322). The absolutes are not comparable to the board.

### 2b. Control 1, near-duplicates (`cross_aspect_nodup.py`)

13.3% of test proteins have a training protein with an **identical** known set, and they
share **87.8%** of their new BP terms. That is copying, not predicting. But the
relationship is **graded**, which a pure duplicate effect would not be:

| top-1 Jaccard of the known set | n | overlap of the NEW BP answers |
|---|---|---|
| identical (1.0) | 176 | **0.878** |
| 0.8 to 1.0 | 130 | 0.682 |
| 0.5 to 0.8 | 399 | 0.413 |
| **< 0.5** | **616** | **0.145** |

Base rate is **0.0026**, so even the distant group sits ~55x above chance.

Excluding near-duplicate neighbours from the transfer AND dropping any test protein that
has one: `C = 0.226`, `S = 0.2886`, **`S - C = +0.0626`**. Two thirds of the headline was
duplicates; a third survived.

### 2c. The real pool, as one feature (`set_level_as_reranker_feature.py`)

The only question that matters for building anything: does it survive next to the 72
features we already carry, on the **real** candidate pool? A/B inside the eval cell
(616,223 rows, 4,455 proteins, split by protein 3,118/1,337; both arms identical except
the added column).

| arm | f_micro_w |
|---|---|
| A the 72 we ship | 0.2704 |
| B + `set_level_transfer` | **0.3600** (`+0.0896`, feature gain share **48.6%**) |

One column, half the model's gain. **A feature that dominates like that is usually a
leak**, and the guard from 2b was the wrong guard: it blocked duplicate proteins, not
time.

### 2d. Control 2, the window artefact (`window_prior_control.py`, `nested_setlevel_over_prior.py`)

The transfer source is the training proteins' **post-t0** BP answers. So the feature can
encode *what this annotation window is curating*, which is a thing that does not exist at
t0. Test: replace it with a **dumb per-window term-frequency prior** that ignores the
protein's known set entirely.

| arm | f_micro_w | vs A |
|---|---|---|
| A the 72 we ship | 0.2714 | |
| A' + **window term prior** (knows nothing about the protein) | 0.3379 | **+0.066** |
| B' + `set_level_transfer` **on top of the prior** | **0.3669** | **+0.029 over A'** |

**Two thirds of the +0.0896 is a trivial temporal artefact.** A counter that never looks
at the protein reproduces it. The window prior is nonzero on 63.3% of rows against
set-level's 14.9%, which is why it carries so far.

**One third survives, with the artefact already inside the baseline: `+0.0290`** (set-level
still takes 39.05% of the gain). That is the number, and it is the only one isolated
against the artefact.

## 3. What this licenses, and what it does not

**Measured and safe to state:**
- The hole is real and structural: LK-BPO's only prior-knowledge channel is one
  first-order counter worth +0.0037, because `anc2vec_query` is mute there by
  construction and `anc2vec_neighbor` is negative.
- The regime pattern: first on NK-BPO ahead of TransFew, third on LK/PK-BPO. The wall is
  about **exploiting prior knowledge**, not about BP.
- Known-set similarity predicts new-BP similarity in a **graded** way, ~55x base rate
  even for proteins overlapping less than half.
- Reading the SET adds **+0.029** over a window-frequency artefact that already sits in
  the baseline, on the real pool, as one feature.

**NOT licensed:**
- **`+0.029` is not a board delta.** Both features are built from post-t0 answers. The
  nesting isolates set structure from the *window prior*, **not from time**. In production
  the transfer would come from the frozen t0 corpus. Read it as an optimistic ceiling.
- The full-vocabulary absolutes (0.4172, 0.9684) belong to an easier candidate regime and
  are not comparable to anything on the board.
- `0.2704` is not the deployed number. It trains on eval-window proteins; the deployed
  recipe trains on v160 to v227 pairs and delivers 0.2131. That ~+0.057 gap is a
  **temporal domain shift**, real but not actionable (training on the test window is
  leakage in production).

## 4. The one experiment that would settle it

Export the t0 corpus (protein to terms, with aspect) for the LK targets and rerun the
nested arm with the transfer sourced from **t0 only**. Then `+0.029` becomes either a real
production lever or nothing. It is a bounded query, not a four-hour export: the benchmark
already freezes knowledge for the eval targets, but covers only **103 of 523** LK-BPO
proteins, which is why tonight's work had to run on PK.

Until then, the honest thesis statement is that the channel is **located and bounded**,
not crossed: we know which cell needs it, why our implementation cannot carry it, and what
it is worth at best.

Receipts: `storage/cooc_experiment/{cross_aspect_channel_ceiling,cross_aspect_nodup,
set_level_as_reranker_feature,window_prior_control,nested_setlevel_over_prior}.{py,json}`.
