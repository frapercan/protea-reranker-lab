# Information Accretion (IA) provenance

LAFA-aligned Information Accretion table for weighting CAFA metrics
(weighted Fmax `wf` and the `S_min` family) comparably to the
functionbench.net leaderboard.

File: `datasets/ia/IA-swissprot-exp-v227.txt`

Format: two-column tab-separated (`GO_term<TAB>IA_bits`), no header.
This is exactly the format consumed by `cafaeval`'s `ia_parser`
(`cafaeval-protea/src/cafaeval/parser.py::ia_parser`), so it can be
passed straight to `cafa_eval(..., ia=<this file>)`.

## Definition

IA is computed exactly as in CAFA / LAFA, by running the `democafa`
package (the same code LAFA uses). The formula is not reimplemented.

Per Clark and Radivojac (2013):

```
IA(t) = -log2 P(t | parents(t))
      = -log2 [ count(proteins annotated with t)
                / count(proteins annotated with ALL parents of t) ]
```

over the propagated reference corpus. Roots have no parents so IA is
defined to be 0. A term whose count equals its parent-intersection
count also yields 0 (no added surprise).

## Reference corpus

| Field | Value |
|-------|-------|
| Corpus | SwissProt (reviewed UniProtKB) GOA annotations |
| GOA release | v227 (Sep 2025), PROTEA annotation_set `5e84d6c5-a104-407d-8c02-b6b653b571b7` |
| Evidence codes | `Experimental,IC,TAS` (democafa default) |
| Resolved codes | EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP, IC, TAS (13 codes) |
| Excluded | IEA, all COMPUTATIONAL (ISS, ISO, ISA, ISM, IGC, RCA), all PHYLOGENETICAL (IBA, IBD, IKR, IRD), NAS, ND, and every `NOT` qualifier |
| Aspects | P, C, F (single-letter, as `democafa` expects) |

### SwissProt restriction (the LAFA-comparability gate)

`democafa` restricts the IA corpus to SwissProt (reviewed) entries.
The PROTEA DB table `protein_go_annotation` is sourced from
`goa_uniprot_all` (all UniProtKB, not SwissProt-only), so a restriction
was required to keep the frequencies comparable to LAFA.

The restriction was applied through the `protein` table, which carries
a `reviewed` boolean (index `ix_protein_reviewed`). Two facts made this
clean (verified, not assumed):

1. Every protein row in the DB has `reviewed = true` (615,960 rows;
   zero unreviewed rows). The `protein` table already is the SwissProt
   set for this deployment.
2. All 88,103 distinct accessions carrying an experimental/IC/TAS
   annotation in v227 join to a `protein` row (zero unmatched), and all
   of those are `reviewed = true`.

So the experimental corpus extracted from v227 is already SwissProt-only.
No silent all-UniProt fallback was used.

Note on DB state: the corpus export and the SwissProt-coverage checks
above were run against the fully populated PROTEA DB. Shortly after the
export completed, the DB was emptied during the session (the recurring
`postgres_data` volume wipe landmine, not triggered by this task), so the
`build_corpus.sql` query cannot currently be re-run for live
re-verification. The IA artefact itself is independent of that: it was
fully computed from the captured corpus before the wipe. Re-running
`build_corpus.sql` after a DB restore reproduces the same corpus.

## Ontology (OBO)

| Field | Value |
|-------|-------|
| File | `datasets/bench-v1-K5/go.obo` (the same OBO used in eval sweeps) |
| `data-version` | `releases/2026-01-23` |
| `format-version` | 1.2 |
| md5 | `74281b5c368c4fcd8d0ee26796750bc4` |
| Edges kept for propagation | `is_a` and `part_of` only, no cross-aspect edges (`democafa.utils.ia.clean_ontology_edges`) |
| Obsolete terms | ignored (`obonet.read_obo(..., ignore_obsolete=True)`) |

OBO choice note: t0 for the leaderboard alignment is GOA v227 (Sep 2025),
but the OBO is the `releases/2026-01-23` snapshot already pinned by the
lab eval sweeps. It was chosen for internal consistency: the IA table,
the propagation in `cafaeval`, and the ground-truth / prediction term
universe must all use the same ontology, otherwise IA can go negative or
terms fall out of the universe. The GO structure changes little over the
roughly four months between Sep 2025 and Jan 2026, and propagation only
uses `is_a`/`part_of` topology. Using the eval OBO is the safer choice
than fetching a Sep-2025 OBO that would mismatch the evaluator.

## Tooling

| Field | Value |
|-------|-------|
| Package | `democafa` (github.com/anphan0828/democafa_package) |
| Commit | `742814eb48740526c5fa14ce133ec35b4f995951` (2026-05-18) |
| Entry point | `python -m democafa.datacollection.propagate_and_ia` (wraps `democafa.utils.ia.run` with `propagate_annotations=True`) |
| Deps | obonet, numpy, pandas, networkx, scipy, requests |

## Reproduction

1. Export the raw SwissProt experimental corpus from the PROTEA DB
   (see `build_corpus.sql` in this directory). Output columns:
   `EntryID`, `term`, `aspect`.
2. Run democafa propagation + IA against the pinned OBO:

```
python -m democafa.datacollection.propagate_and_ia \
  --terms   swissprot_exp_v227_raw.tsv \
  --graph   datasets/bench-v1-K5/go.obo \
  --tsv_propagated swissprot_exp_v227_propagated.tsv \
  --output_tsv     datasets/ia/IA-swissprot-exp-v227.txt
```

Propagation expands annotations to all ancestors per aspect before
counting, which is required for the conditional-probability counts.

## Row counts

| Stage | Count |
|-------|-------|
| Raw experimental SwissProt annotations (v227, deduplicated, NOT excluded, obsolete dropped) | 547,133 |
| Distinct proteins | 88,103 |
| Distinct terms (raw) | 26,609 |
| Propagated annotations (to roots, per aspect) | 3,590,898 |
| Distinct terms (propagated) | 29,486 |
| IA rows written (full OBO term universe across P, C, F) | 38,739 |
| IA rows with nonzero value | 28,391 |

Raw annotations by aspect: P 250,514, C 161,044, F 135,575.

## Validation

| Check | Result |
|-------|--------|
| Negative IA values | 0 (propagation is consistent with the OBO) |
| Roots (GO:0008150, GO:0005575, GO:0003674) | IA = 0.0 exactly |
| IA range | 0 to 15.9462 bits |
| Mean IA | 2.773 bits |
| Rarest terms | highest IA (singletons at about 15.9 bits) |
| Per-aspect coverage | present for P, C, F (see below) |

Per-aspect IA over terms seen in the corpus:

| Aspect | terms | min | max | mean | median | nonzero |
|--------|-------|-----|-----|------|--------|---------|
| C | 2,959 | 0.000 | 14.894 | 3.532 | 2.485 | 2,577 |
| F | 7,283 | 0.000 | 14.924 | 3.763 | 3.442 | 6,666 |
| P | 19,244 | 0.000 | 13.828 | 1.778 | 0.485 | 13,364 |

Spot checks (general terms low IA, predictable-from-parent terms near
zero, rare terms high), all sane:

| Term | Meaning | IA (bits) |
|------|---------|-----------|
| GO:0003824 | catalytic activity (shallow MF) | 1.499 |
| GO:0016787 | hydrolase activity | 1.750 |
| GO:0005634 | nucleus | 0.954 |
| GO:0005737 | cytoplasm | 0.347 |

IA is conditional surprise given parents, not raw depth, so it does not
correlate monotonically with topological depth by construction; the
extremes (roots at 0, singletons near the per-aspect max) confirm
correct behaviour.
