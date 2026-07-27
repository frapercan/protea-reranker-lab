# research/ - preserved experiment procedures

These are the research procedures that produced the campaign's measurements.
They ran from a scratch directory outside every repository, so none of them was
under version control. The machine they lived on is being reformatted, and an
audit of where each stage of the champion pipeline lives classified them as
capabilities that would die with it: the outputs were archived, the procedures
were not.

They are preserved **as they ran**. Nothing here was reformatted, renamed or
tidied, because the value of a preserved procedure is that it is the procedure
that produced the number. Expect hardcoded paths, inline constants, and scripts
that assume a live database. Treat them as evidence first and as code second.

Nothing here is on any CI path: the quality gates read `src`, `scripts`,
`tests` and `experiments`.

## Why each group was kept

**Load-bearing for published claims.** These regenerate numbers that appear in
the thesis, the interface or the Sphinx book, so losing them would leave a
published claim with no regenerator.

- `regen_headline/` - the nine-cell board table behind the 7-of-9 claim, the
  derivation of the sealed headline, the pre-t0 vocabulary claim, paired
  bootstrap CIs, the ProtST A/B decision harness, and the BP frontier
  characterisation that is the Pillar 4 contribution.
- `fullgo_models/` - the M2 classifier lineage, including the only code that
  persists an M2 checkpoint, and the comparative booster trainers behind the
  feature-family governance.
- `layer_ablation/` - the board-faithful KNN GO-transfer confirm harness that
  every representation study reuses.
- `feature_necessity/`, `length_strat/`, `text_scorer/` - the stratification and
  necessity analyses. Three-axis stratification is a standing norm of the
  project and had no implementation anywhere in versioned code.

**Refuted, and kept deliberately.** A refutation whose code is gone cannot be
defended. These negative results characterise the calibration wall and are a
contribution in their own right.

- `cooc_experiment/`, `struct_gate/`, `phylo_profile/`, `string_v12/`,
  `protex/`, `consensus/`, `transfew_calib/`, `deepgose_rescore/`,
  `joint_model/`, `entail_kwta/`.

**Capabilities the author may want again.**

- `sse_poc/`, `sse_full/`, `sse_evidence/`, `sse_kwta_gen/` - sparse
  semantic-entailment: GO EL normal forms as soft sparse-containment penalties,
  soft k-WTA with a straight-through estimator, seed consensus. Refuted at the
  board, but an intact differentiable-ontology capability.
- `kwta_go_encoder/` - the learned sparse GO encoder chain with separability
  hard negatives and an IA-weighted InfoNCE sequence head.
- `track_b_backfill/` - the typed-column backfill whose Alembic revision is
  versioned while its data migration was not.

## Candidates for promotion to platform operations

Preservation is not integration. These are the ones worth promoting, in rough
order of value, because they are either standing norms or gates that every
future result depends on:

1. **Stratification** (length x category x neighbour identity). A standing norm
   with no versioned implementation. Every stratified table in the campaign was
   made by an ad-hoc script.
2. **The board-frame recipe as a versioned fixture.** The mechanism is already a
   platform operation; the parameterisation that reproduces the published number
   is prose in a runbook.
3. **The control family** (shuffled feature, fixed score, random order, matched
   volume, IA precision). This is what separates a result from a story, and it
   was reimplemented per experiment.
4. **The cross-fit gate** (sweep on nine folds, measure on the held-out tenth,
   report mean and fold spread). Deterministic protein folding is already a
   platform payload field; the gate around it is not.
5. **Paired bootstrap CIs in the board frame.** Two versioned implementations
   exist, but both bootstrap Fmax rather than the headline metric, and neither
   runs inside the board frame.
