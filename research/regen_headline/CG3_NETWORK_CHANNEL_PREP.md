# CG.3 -- Network channel: DESIGN + FEASIBILITY (no download)

**Scope.** Design and feasibility ONLY, up to but NOT including any external download. The download
needs explicit author greenlight (section 5). This slice answers one question before spending
bandwidth: *are the LK-BPO / PK-BPO target proteins even in a pre-t0 biological network, with
annotated partners that could propose their missing BP tail?*

**Feasibility verdict: STRONG GO.** 97.8% of the 4,925 target proteins are in tier-1,
densely-curated model-organism interactomes (all STRING v12.0 reference proteomes); STRING mapping is
effectively universal for this target set; the binding constraint (">=1 t0-annotated partner") is
near-satisfied by construction for the tier-1 fraction. The download is worth requesting.

Rationale for the channel (from CG.1): the missing true BP tail is *reachable* (pool misses only
26-29% of true IA-mass) but not *separable* from the false tail by any in-house sequence-KNN +
intra-organism co-occurrence signal (generation precision ~0.4-1.6% IA). BP is hypothesised to be a
NETWORK property (process = who you interact / co-express / co-occur with), an axis orthogonal to the
sequence-only closure we have exhausted. CG.3 tests whether a protein's already-annotated *network
partners* propose its about-to-be-gained BP terms at a precision that beats the 1% co-occurrence bar
and approaches the 11.59% classifier bar.

---

## 1. Exact resource spec (all releases predate t0 = 2025-09-04)

The t0 leakage boundary: GOA v227 / obo releases/2025-07-22 / annotation cutoff **2025-09-04**; test
window runs v227->v230 (2026-03-04). Every network edge set AND every partner annotation used must be
frozen at or before 2025-09-04.

### Recommended primary: STRING v12.0 (PPI / functional association)
- **Version / date:** v12.0, current since **2023-07-26** (>2 yr before t0 -- edge freeze is provable
  by release date; STRING has no interim re-release of v12.0).
- **Scale:** 59,309,604 proteins across **12,535 organisms**, 27.5B interactions.
- **URL:** `https://stringdb-downloads.org/download/` (per-organism files under
  `protein.links.v12.0/` and `protein.links.detailed.v12.0/`; UniProt mapping under
  `protein.aliases.v12.0/`).
- **Files needed:** per-organism `<taxon>.protein.links.detailed.v12.0.txt.gz` (edges +
  per-channel scores) and `<taxon>.protein.aliases.v12.0.txt.gz` (STRING<->UniProt map). We need only
  the ~22 tier-1 organisms (covering 97.8%) plus optionally the ~90 tail taxa.
- **Approx size:** per organism 20-400 MB for `links.detailed`, ~5-20 MB for aliases; the ~22 tier-1
  organisms total roughly **5-12 GB**. (The all-species `protein.links.full.v12.0.txt.gz` is ~90 GB
  -- NOT needed; per-organism is the efficient path.)
- **License:** data under **CC BY 4.0** (free for academic and commercial use with attribution).
- **Why primary:** best coverage (all reference proteomes), UniProt alias file solves the mapping
  problem, and the `detailed` file lets us gate on the *experimental* + *database* channels only
  (dropping `textmining`, which risks encoding literature that postdates t0 -- see leakage plan).

### Alternatives / corroboration (all pre-t0)
- **BioGRID 4.4.248** (monthly, ~**2025-09-01**, just before t0; archive
  `https://downloads.thebiogrid.org/BioGRID/Release-Archive/`). Curated physical + genetic
  interactions, **MIT license**. Use as an experimental-only, curation-dated cross-check of STRING's
  experimental channel; smaller (~hundreds of MB). BioGRID's dated releases give a *cleaner* edge
  freeze than STRING's channel scores.
- **IntAct** (EBI, **monthly** release; take the snapshot dated <=2025-09-04 from
  `ftp.ebi.ac.uk/pub/databases/intact/`). PSI-MI TAB, **CC BY 4.0 / Apache-2.0**. Curated,
  experiment-backed; overlaps BioGRID/STRING-experimental.
- **Co-expression -- COXPRESdb v8** (data snapshot **2022-06-30**, pre-t0;
  `https://coxpresdb.jp/download/`). Animal only (~11 species: covers human/mouse/rat/fly/zebrafish/
  worm here, i.e. the bulk of tier-1 animals). Orthogonal modality to physical PPI. STRING v12.0's own
  co-expression channel (VAE-based, in `links.detailed`) is the simpler in-band substitute and already
  frozen at 2023-07-26.
- **Phylogenetic profiles -- eggNOG 6.0** (published **Nov 2022**, 12,535 organisms;
  `http://eggnog6.embl.de/download/`) or **OrthoDB v11** (2021, pre-t0). Gives orthogroup membership
  for a presence/absence profile modality. Heavier to engineer; defer unless PPI + co-expression
  underperform.

**Decision:** start with **STRING v12.0 per-organism `links.detailed` + `aliases`** for the ~22
tier-1 organisms. It is one resource, one release date (2023-07-26, provably pre-t0), carries three
network modalities in-band (experimental, database, co-expression channels), and ships its own UniProt
map. BioGRID 4.4.248 is the curation-dated cross-check.

---

## 2. Coverage feasibility -- the decider (computed NOW, on-disk data only)

**Method.** Target accessions: aspect=P rows of `groundtruth_{LK,PK}.tsv` in
`repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt/` -> 523 LK + 4,402 PK = **4,925
unique**. Taxonomy resolved offline by streaming the on-disk pre-t0 GOA GAF
`storage/gaf_cache/goa_uniprot_all.gaf.225.gz` (GOA **v225**, pre-t0) and reading the taxon column
(col 13) per accession -- no download, no DB. **4,922 / 4,925 matched** (3 are proteins new enough to
carry no v225 GOA annotation). Receipts: `cg3_extract_taxa.sh`, `cg3_acc_taxon.tsv`, `cg3_coverage.json`.

**Taxon distribution (the coverage number).** The target set is overwhelmingly classic
model-organism proteins -- exactly the taxa STRING covers best:

| cell | targets | resolved | tier-1 model-org | tier-1 frac | tail | distinct taxa |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| LK-BPO | 523 | 521 | 511 | **98.1%** | 10 | 22 |
| PK-BPO | 4,402 | 4,401 | 4,305 | **97.8%** | 96 | 98 |
| overall | 4,925 | 4,922 | 4,816 | **97.8%** | 106 | 104 |

Top taxa (both cells): *M. musculus* (1,263), *H. sapiens* (1,194), *S. cerevisiae* (508), *R.
norvegicus* (508), *A. thaliana* (436), *S. pombe* (302), *E. coli* K-12 (174), *D. melanogaster*
(172), *D. rerio* (48), *C. albicans* (41), *M. tuberculosis* (37), *O. sativa* (35). All are STRING
v12.0 reference proteomes with dense experimental + curated interactomes.

**Two-part coverage argument.**
1. *Mappable into the network id-space?* -- **~100%.** STRING v12.0 is built from the reference
   proteomes of 12,535 organisms and ships a UniProt alias file; every tier-1 organism here is a
   STRING reference proteome, so essentially every target accession maps. Even the 2.2% tail taxa
   (fungi: *Aspergillus*/*Neurospora*; plants; a few viruses incl. SARS-CoV-2 2697049;
   non-model bacteria) are almost all among STRING's 12,535 genomes -- mapping loss is a rounding error.
2. *Has >=1 t0-annotated partner?* -- **near-total for the 97.8% tier-1 fraction.** Model-organism
   STRING networks are dense (e.g. human ~19.6k nodes, near-fully connected at medium confidence), so
   ">=1 partner" is effectively guaranteed; and those partners are heavily GO-annotated at t0 (the
   frozen `reference_annotations.parquet` holds 5.88M annotations over 574,627 proteins). The precise
   per-protein "partner with >=1 frozen BP annotation" count is only computable once STRING edges are
   on disk (it needs the edge list) -- that is the first check to run *after* download, but the taxon
   profile already makes a low-coverage failure implausible for tier-1.

**Honest read on the tail.** The ~2.2% (106 proteins, ~90 taxa) in non-model organisms will have
sparser STRING interactomes and less-annotated partners; expect degraded but non-zero coverage there.
This does not threaten the slice: the LK/PK cells are 98%/98% tier-1, so the channel is evaluated
where it is strongest. **No taxon-coverage NO-GO.**

---

## 3. Leakage plan (freeze edges + partner annotations at t0, and prove it)

**One line:** edges from STRING v12.0 (release 2023-07-26, provably pre-t0) restricted to the
experimental+database(+coexpression) channels; partner *annotations* drawn ONLY from the frozen
`reference_annotations.parquet` (v227 = t0); partners that are themselves test targets contribute only
their t0-known set, never their gains.

**Freezing procedure.**
1. **Edge freeze (by release date).** STRING v12.0 static release 2023-07-26 << t0 2025-09-04. Drop
   the `textmining` channel (it can encode literature co-mentions that postdate the v12.0 corpus in
   spirit and are the weakest-provenance channel); keep `experimental`, `database`, `coexpression`,
   optionally `neighborhood/fusion/cooccurrence` genomic channels. Cross-check the retained edges
   against curation-dated BioGRID 4.4.248 (<=2025-09-01) for the experimental channel.
2. **Partner-annotation freeze.** A partner p' proposes a BP term t only if p' carries t (propagated)
   in the frozen t0 corpus `reference_annotations.parquet` (cutoff 2025-09-04) -- never from current
   GOA. This is the SAME artifact CG.1 proved clean.
3. **Self / target-partner exclusion.** Exclude self-edges. If a partner p' is itself a test target,
   use only p''s frozen *known* set (PK: `groundtruth_PK_known.tsv`; LK: reference F/C -- but note LK
   partners contribute BP only if they have frozen BP, which by the LK definition they may not), never
   its post-t0 gains. This blocks target-to-target gain leakage through shared edges.

**Empirical proof (analogous to CG.1's "0 BP terms in LK known sets").**
- (a) *Edge date:* assert the STRING release string == v12.0 / 2023-07-26 and that all retained edges
  carry no post-t0 provenance (channel-score check; textmining dropped).
- (b) *Annotation freeze, the decisive check:* re-run the whole generator with the STRICTLY-EARLIER
  **v225** GOA GAF (already on disk, pre-t0) as the partner-annotation source and confirm the proposed
  (protein, BP-term) set is a **subset** of the v227 run with no term that is a post-t0 gain of the
  target -- i.e. **0 proposals equal to a held-out target gain sourced from a post-t0 partner
  annotation**. Zero is the pass, exactly as CG.1 showed 0 BP terms leaked into LK known sets.
- (c) *Round-trip:* verify no target's own post-t0 gained term enters via a partner that is a copy of
  the target (dedup on sequence-identical / same-accession partners).

---

## 4. Generator + measurement design (same §4.3 metric as CG.1/CG.2)

**Generator.** For target protein p with frozen network partners N(p) (STRING edges above a confidence
cut, e.g. combined_score>=400 on the retained channels), each partner p' carries its frozen t0 BP
annotation set B(p') (propagated ancestors-or-self, BP-restricted, from `reference_annotations`). The
proposal score for a candidate BP term t NOT already in p's deployed pool:

  s(t | p) = sum over p' in N(p) of w(p, p') * 1[t in B(p')]

with w(p, p') a monotone function of edge confidence (start: w = combined_score/1000; ablate w=1
unweighted, and a degree-normalised variant s/|N(p)| to control for hub inflation). This is pure
GENERATION: every scored (p, t) is a pair absent from p's pool, identical in spirit to CG.1's
PPMI(seed->BP) generator but with *network partners* as the seed set instead of the protein's own
known annotations.

**Recall-added metric (the §4.3 number, identical to CG.1/CG.2).** Propagate proposals and ground
truth with the lab `lafa_t0_Sep_2025` obo/IA. At each proposals-per-protein k in {5,10,25,50},
accumulate, micro over proteins, the IA-mass ADDED that is TRUE (ancestor-or-self of a gained gt BP
term, minus already-covered mass) vs ADDED that is FALSE, and report **generation precision = added-true
/ (added-true + added-false)** and **fraction of the FN IA-mass recovered**. Bars to beat (from CG.1):
in-house co-occurrence **~1.0-1.6%** (the floor to clear) and the full-GO classifier **11.59%** (the
target to approach). Ceiling per cell is the FN IA-mass the pool misses: LK 2,399 (26.1%), PK 22,924
(28.6%). Reuse `cg1_step1_gate.py`'s obo/IA/propagation/IA-mass machinery verbatim; swap only the
seed source (partners' B(p') for the protein's own K_p).

**Gate.** If network-partner generation precision clears the ~1% floor by a clear margin AND recovers
a materially larger FN fraction than co-occurrence at matched k, proceed to Step 2 (union +
cafaeval in the reconstructed board frame with `-known` per `BOARD_FRAME_RECONSTRUCTION.md`).
Otherwise NO-GO, and the SEPARABILITY wall is confirmed to extend to the network axis.

**Temporal-transfer gate.** The generator is a t0-only statistic (edges pre-t0, annotations t0), so it
is deployable by construction. Confirm transfer the way the campaign requires: stratify the added-true
precision by (i) length x category(NK/LK/PK x aspect) x neighbor-identity, (ii) taxon (tier-1 vs tail),
(iii) partner-degree, with bootstrap CIs on the deltas; and check that IEA-only partner annotations
still transfer (IEA-is-signal, per the annotation-RAG finding). A lineage/hub control -- precision must
not come only from high-degree hubs or from same-taxon partners -- guards against the network encoding
taxon membership rather than process.

---

## 5. The explicit ASK (one-paragraph greenlight)

**Requesting greenlight to download STRING v12.0 (release 2023-07-26, provably pre-t0), per-organism
files only, for the ~22 tier-1 model organisms that cover 97.8% of the 4,925 LK-BPO+PK-BPO targets:
`<taxon>.protein.links.detailed.v12.0.txt.gz` + `<taxon>.protein.aliases.v12.0.txt.gz`, roughly 5-12 GB
total, CC BY 4.0.** This is the only external data CG.3 needs; everything else (partner annotations,
obo, IA, metric machinery, taxon profile) is already on disk and frozen at t0. Coverage feasibility is
already settled from on-disk GOA (97.8% tier-1, ~100% STRING-mappable), so the download risk is low and
the first post-download step is the concrete ">=1 t0-annotated partner" count. Optional add-on if you
want a curation-dated experimental cross-check: BioGRID 4.4.248 (~2025-09-01, MIT, a few hundred MB).
**No download proceeds without your yes.**

---

## Receipts (all under `storage/regen_headline/`)
- `CG3_NETWORK_CHANNEL_PREP.md` -- this file.
- `cg3_coverage.json` -- per-cell taxon coverage summary (the decider number).
- `cg3_acc_taxon.tsv` -- accession -> taxon for all 4,922 resolved targets.
- `cg3_extract_taxa.sh` -- the offline GOA-v225 taxon-extraction pass (no download, no DB).
- Reuses: `cg1_step1_gate.py` (metric machinery), `BOARD_FRAME_RECONSTRUCTION.md` (`-known` board frame),
  `storage/protea-frozen-v227-2025-09-04/reference_annotations.parquet` (frozen t0 partner annotations),
  `protea-lafa-knn/lafa_t0_Sep_2025/{go-basic.obo,IA.tsv}`.
