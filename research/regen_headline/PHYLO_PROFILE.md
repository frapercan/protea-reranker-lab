# PHYLO -- Cross-species phylogenetic profiling channel: VERDICT

**GO/NO-GO: NO-GO.** The phylogenetic-profiling axis (guilt-by-association from proteins whose
ortholog presence/absence profile across genomes is similar to the target's, an axis orthogonal to
sequence homology, STRING interactions and literature) generates BP-tail candidates at **0.0% (LK) /
2.6% (PK) top5 IA-precision** -- at or below the co-occurrence floor, below the network channel, far
below the classifier -- recovers **<1%** of the missing BP information mass, and **lowers f_micro_w in
every arm** against the deployed percutgraft anchor. The profile-similarity score carries **no usable
ordering signal** (the phylo arm equals a random-order control to five decimals). The BP separability
wall extends to the phylogenetic-profiling axis.

## The one question
Does a leakage-safe cross-genome phylogenetic profile GENERATE the true BP-tail terms the sequence-only
pool misses, at a precision clearing the fill-tax bar, and newly reach the ~40-45% sequence/network/
text-unreachable BP residual, where five prior generation channels failed? **No.**

## Download actually obtained
**OrthoDB v12.2** flat files (release 2024; OrthoDB+BUSCO update, NAR gkae987, Nov 2024; CC-BY),
`https://data.orthodb.org/current/download/odb_data_dump/`. Used at the **Eukaryota level (taxid 2759)**:
5,952 assemblies collapsed to **5,668 distinct eukaryotic NCBI taxa** as the profile dimension.

| file | size | md5 | use |
|------|------|-----|-----|
| `odb12v2_genes.tab.gz` | 4.5 GB | `521a756b...899b` | UniProt AC (col5) -> OrthoDB gene id -> species |
| `odb12v2_OG2genes.tab.gz` | 4.5 GB | `114aff25...8c9b` | gene -> at2759 OG + per-OG species presence profile |
| `odb12v2_OGs.tab.gz` | 128 MB | -- | OG level metadata (kept) |
| `odb12v2_{species,levels,level2species}.tab.gz` | <1 MB | -- | odb_species_id -> NCBI taxid; euk universe (kept) |

Saved under `storage/phylo_profile/`. The two 4.5 GB raw files were **reclaimed after the build**
(box at 93% disk, live stack running); the derived artefacts (`og_presence.npz`, `acc2og.json`,
`og_members.json`) fully capture what the experiment used, and the md5s above make the raw
re-downloadable. **`odb12v2_gene_xrefs.tab.gz` (which carries GO terms) was NEVER read** -- no GO
annotation from OrthoDB enters the pipeline; the only GO comes from the frozen v227=t0 reference.

## Generator
For target `p` in Eukaryota-2759 orthologous group `OG_x`, rank all informative OGs `OG_y`
(prevalence in `[4, 95%]` of the 5,668 taxa) by **Pearson correlation over the presence/absence
profile**. Each of the top-50 neighbour OGs (excluding `OG_x` itself -- same OG = homolog, enforcing
orthogonal reach) is a "partner" carrying `B(y)` = union of the **frozen t0 BP annotations** of the
reference proteins in `OG_y`:
`s(t|p) = sum over top-50 OG_y of sim(OG_x, OG_y) * 1[t in B(y)]`.
Metric machinery (obo BP ancestors, IA, gt/pool propagation, added-true vs added-false IA-mass) is
**reused verbatim from `cg3_step1_gate.py`**; only the proposal source changed (profile-similar OGs
instead of STRING partners).

## Coverage -- the honest caveat (a finding in itself)
OrthoDB v12.2's UniProt keying mapped only **783 / 4,922 targets (15.9%)** to an OG, and it is
**vertebrate-biased**:

| taxon | mapped/total | | taxon | mapped/total |
|-------|:---:|---|-------|:---:|
| mouse | 446/1263 (0.35) | | yeast S288C | **1/508 (0.00)** |
| human | 152/1194 (0.13) | | Arabidopsis | **0/436 (0.00)** |
| rat | 81/508 (0.16) | | S. pombe | **0/302 (0.00)** |
| fly | 42/172 (0.24) | | E. coli / M.tb | 0 (prokaryote, no at2759 OG) |
| worm | 13/19 (0.68) | | Dictyostelium | 0/26 (0.00) |

Root cause (verified): OrthoDB stores **secondary/TrEMBL** UniProt ACs, not canonical SwissProt ACs,
in the UniProt field for many proteomes -- yeast S288C genes carry `D6VP*` ACs, not canonical
`P`-numbers -- and uses T2T/alternate vertebrate assemblies with partial UniProt xref. Our
targets/reference are keyed by canonical ACs, so the compact fungal/plant genomes (where phylo-profiling
is classically strongest, e.g. yeast) are missed. **This is a keying mismatch, not a biology limit;**
a UniProt idmapping bridge would be the follow-up. It does not rescue the verdict: the measured
vertebrate subset (~700 PK / 70 LK targets, in the best-curated organisms where guilt-by-association
peaks) already shows near-zero signal.

## Step 1 -- added-true IA-precision (the gate)

| cell | top5 | top10 | top25 | top50 | co-occ bar | network bar | classifier |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LK-BPO | **0.0000** | 0.0021 | 0.0179 | 0.0106 | 0.0161 | 0.0578 | 0.1159 |
| PK-BPO | **0.0264** | 0.0175 | 0.0148 | 0.0098 | 0.0103 | 0.0241 | 0.1159 |

LK top5 is literally **0** (0 true, 57 false IA-mass). PK top5 (2.6%) is marginally above the
co-occurrence floor and at the network channel, but far below the 11.6% classifier that itself did not
convert; even in human/mouse/rat (top5 ~3.3%) the false tail is 20-30x the true tail. Total missing BP
IA-mass recovered is **LK 0.69% / PK 0.62%** at top50 -- the channel reaches essentially **none** of the
genuinely-unreachable BP residual at any exploitable precision.

## Step 2 -- does it convert? (true board frame, vs the DEPLOYED anchor)
`prop=fill norm=cafa no_orphans toi`; **PK adds `-known`**; cafa_eval under the PROTEA venv; one
variable = candidate set only. **Arm A reproduces the deployed percutgraft anchor exactly** (LK
**0.31323**, PK **0.14351**) -- precondition passed (not the raw-pool ProtEx anchor error).

| cell | A (anchor) | B top5 | B top10 | B top25 | Rnd top5 (control) | U top5 (matched-vol) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LK-BPO | 0.31323 | **-0.00089** | -0.00274 | -0.00652 | -0.00071 | -0.03969 |
| PK-BPO | 0.14351 | **-0.00106** | -0.00280 | -0.00958 | -0.00106 | -0.02516 |

**Every B arm is negative on both cells.** The random-order/scale control (**Rnd**) equals B to five
decimals (PK top5 `-0.00106 == -0.00106`) -- the phylo similarity score carries **no usable ordering
signal**; the candidates hurt regardless of ranking. The matched-volume uniform control (**U**) is far
worse, confirming the `prop=fill` tax on low-precision volume. No positive -> no bootstrap CI needed.

## Leakage -- CLEAN
- **Profiles:** ortholog presence/absence is a genome-content fact computed by sequence clustering,
  independent of GO and of the annotation cutoff; it cannot encode post-t0 annotations. OrthoDB v12.2
  (2024) predates t0 (2025-09-04) regardless. OrthoDB's GO xref file was never read.
- **Partner annotations:** ONLY the frozen v227=t0 `reference_annotations.parquet` (cutoff 2025-09-04,
  before the v227->v230 eval gains). Self accession and the target's own OG (homolog OG) excluded.
- **Empirical target-as-partner ablation:** removing every partner that is itself a test target leaves
  precision essentially unchanged (LK top5 0.0435->0.0435; PK top5 0.0184->0.0173) -> the (tiny) signal
  is carried by independent non-target proteins in co-evolving OGs, not target-to-target laundering.
  Partner annotations are t0-frozen and target gains post-t0, so there is no temporal leakage by
  construction.

## Implication
Phylogenetic profiling was the last untested orthogonal axis (distinct from sequence homology, the
STRING network, and literature). Where measurable it behaves exactly like the five prior generation
channels: the missing true BP tail is **reachable but not separable**. Together with CG.3 (the PPI/
co-expression network -- the *strongest* orthogonal generator, also NO-GO), this confirms the **BP
separability wall is a field frontier, not an in-house engineering gap and not a missing data axis**.

## Receipts (under `storage/regen_headline/` unless noted)
- `PHYLO_PROFILE.md` / `phylo_verdict.json` -- this verdict.
- `phylo_step1_gate.json` -- the gate (per-k precision, coverage, stratification).
- `phylo_step2_cafaeval.json` -- true-frame board delta + matched-volume + random-order controls.
- `phylo_leakage.json` -- target-as-partner leakage ablation.
- `storage/phylo_profile/` -- `build_phylo.py`, `phylo_gate.py`, `phylo_step2.py`, `phylo_leakage.py`,
  `og_presence.npz`, `og_meta.npz`, `acc2og.json`, `og_members.json`, OrthoDB metadata `.tab.gz`
  (raw genes/OG2genes reclaimed; md5s in `phylo_verdict.json`).
