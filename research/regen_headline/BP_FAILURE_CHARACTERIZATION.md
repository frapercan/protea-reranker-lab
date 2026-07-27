# The shape of our Biological-Process failure: LK-BPO and PK-BPO characterized

Read-only, frozen-data, board frame. Machinery reuses `cooc_experiment/atlas_rebuild.py`
(cafaeval propagation + gt-membership pool oracle + IA-weighted per-protein micro-F
decomposition). Script `bp_failure_atlas.py`; per-protein JSON
`bp_atlas_perprotein_{lk,pk}.json`; tables `bp_failure_atlas.json`; network features
`bp_string_features.json`. This is the first time LK-BPO is characterized at this depth
(PKW.3 was PK only), and the first time PK-BPO is characterized in the board `-known` frame.

## Frame and gate (both cells reproduce the deployed headline)

| cell | n proteins | delivered f_micro_w | anchor | pool oracle f | global gap |
|------|-----------:|--------------------:|-------:|--------------:|-----------:|
| LK-BPO | 523 | 0.3113 | 0.3110 | 0.7867 | 0.4754 |
| PK-BPO | 4397 | 0.1413 | 0.1402 | 0.6046 | 0.4633 |

Delivered = the deployed percut+graft predictions (`.../percut_rerank/predictions/{lk,pk}`),
scored in the board frame: toi restriction, `prop=fill`, IA weights, and for PK the per-protein
`-known` exclusion (a known-true term is neither TP nor FN; a prediction on it is not FP). Oracle =
the best ordering of the retrieval+classifier POOL (every pool term that is a propagated-true term
placed on top). Headroom_i = the marginal move in the global delivered f when protein i alone is
served by the pool oracle (tau frozen at the delivered global optimum). Marginal headrooms do NOT
partition the gap (micro-F is nonlinear). PK n is 4397 not 4402: five proteins have all BP terms
already known in t0, so the board frame leaves them nothing to score.

## The one-line answer

Our BP failure is **not a property of the proteins we fail on** (they are ordinary in length, taxon,
and homology) and **not a ranking bug the deployed ranker can fix on its own**. It is a **retrieval /
generation** failure: the true BP terms are disproportionately **not in the candidate pool at all**,
and this is far more severe for PK (2/3 of the missed IA-mass is unreachable) than for LK (~1/2).
The single axis that separates the proteins we get right from the ones we get wrong, in both cells,
is **pool recall** (is the true term even a candidate), not any intrinsic protein property.

---

## The six questions, answered plainly

### 1. Long or short proteins? LENGTH IS NOT THE AXIS.
- PK-BPO: **flat** (Spearman rho of length vs delivered-f = **-0.038**; headroom-per-protein 0.90-1.06
  x1e-4 across every length band). Confirms PKW.3.
- LK-BPO: **weak** negative (rho **-0.113**); longer proteins deliver slightly worse
  (delivered-f 0.38 at <200aa -> 0.19 at >=1200aa) but carry only marginally more headroom.
  A mild multidomain-hardness effect, not a lever.

### 2. Do they have neighbours / network partners?
- **Sequence homologs:** the split is sharp between cells. LK-BPO delivered-f falls with
  neighbour distance (rho **-0.319**; twilight-zone LK proteins are genuinely harder). PK-BPO is
  **flat** (rho **-0.021**): PK proteins essentially ALL have a close homolog (99.7% have a
  "close"-or-better donor; median-distance right-vs-wrong 0.264 vs 0.275), so homology does not
  discriminate success. Having a close neighbour is necessary but nowhere near sufficient.
- **Network partners (STRING v12.0, mouse+rat subset):** the failing proteins are NOT
  network-peripheral and NEVER lack annotated partners. **0% of failing proteins have zero
  t0-annotated partners in either cell** (median clean-channel degree ~700; every failing protein
  sits in a dense, already-annotated interactome). For PK-BPO, degree is **flat** vs delivered-f
  (rho -0.064) and vs headroom (+0.04): being a network hub vs a peripheral node does not predict
  PK-BP success. For LK-BPO, degree correlates positively with delivered-f (rho +0.391) but this is
  **confounded** (degree-vs-pool-recall rho +0.28): high-degree LK proteins are simply the
  better-studied ones with fuller pools, not an independent network lever. See the network section.

### 3. Specific taxa (sparse / exotic)? NO -- the loss is spread across model organisms.
Headroom-per-protein is **essentially flat across taxa** (LK 5.4-11.9 x1e-4, PK 0.5-1.4 x1e-4),
and headroom-share tracks each taxon's share of the target set. The loss sits in the densely-curated
model organisms (mouse, human, rat, arabidopsis, the two yeasts, worm, fly, E. coli), NOT in sparse
or exotic taxa. Delivered-f does vary by organism (PK: human 0.23 > mouse 0.13 > rat 0.09) but that
variation is confounded by pool recall and prior-knowledge structure, not an independent taxon
effect. This is the opposite of the network signal's concentration (CG.3 found network signal in
human/rat and ~0 in sparse taxa); the BP FAILURE is taxon-uniform.

### 4. Recall (terms missing) or ordering (terms present, mis-scored)? THE KEY LK-vs-PK CONTRAST.
Split of the missed true IA-mass into "in the pool but mis-ranked" (ordering; a better ranker could
fix it) vs "not in the pool at all" (recall/generation; no ranker can fix it):

| cell | missed true IA | ordering (in pool) | recall (not in pool) | % ordering | % recall | proteins unreachable |
|------|---------------:|-------------------:|---------------------:|-----------:|---------:|---------------------:|
| LK-BPO | 6,203 | 3,014 | 3,189 | **48.6%** | **51.4%** | 33/523 (6.3%) |
| PK-BPO | 41,337 | 13,886 | 27,451 | **33.6%** | **66.4%** | 1,489/4,397 (33.9%) |

- **PK-BPO is a generation wall:** two-thirds of the missed mass is for terms the retrieval+classifier
  pool never proposes, and 34% of PK proteins have at least some true IA entirely outside the pool.
  This is exactly the "reachable but not separable" / "generation is the only lever" picture from the
  campaign, now quantified in the board frame.
- **LK-BPO is balanced:** roughly half the missed mass is misordering (in-pool, a ranking problem in
  principle) and half is missing terms. LK has the cross-aspect MF/CC context and much higher pool
  recall (only 6.3% of proteins unreachable), so its ceiling (oracle 0.79) is closer and half of the
  gap is, in principle, a ranking/calibration problem rather than a retrieval one.

### 5. A concentrated attackable subpopulation, or diffuse? MOSTLY DIFFUSE, with one structured sliver.
Headroom is moderately concentrated (top-20% of proteins carry 49% of marginal headroom in LK, 66%
in PK, vs ~20% for a random 20%), but that concentration **tracks pool richness, not a coherent
protein class**: the high-headroom proteins are the ones with LARGE, RICH pools where the oracle is
near-1.0 and the deployed ranker mis-orders (the misordering sliver). The dominant PK loss -- the
recall failures -- is diffuse across taxa, length, and homology. So the one thing a targeted method
could attack is the **misordering sliver** (high-pool-recall proteins: LK >=0.75 pool-recall carries
71% of headroom at oracle 0.99 vs delivered 0.40), but note the campaign already found that this
sliver is capped by cafaeval's cross-protein calibration structure -- a better per-protein ranker
does not convert it. The genuinely large PK bucket needs generation, and it is not a subpopulation.

### 6. What distinguishes RIGHT from WRONG BP proteins? POOL RECALL / REACHABILITY -- and a knowledge inversion.
Contrasting the top-delivered-f quartile (RIGHT) with the bottom quartile (WRONG):

| feature | LK right | LK wrong | PK right | PK wrong |
|---------|---------:|---------:|---------:|---------:|
| pool_recall | 0.94 | 0.45 | 0.76 | 0.29 |
| frac unreachable | 0.00 | 0.25 | 0.02 | 0.49 |
| oracle_f | 0.96 | 0.51 | 0.82 | 0.33 |
| median nbr distance | 0.23 | 0.40 | 0.26 | 0.28 |
| length | 396 | 528 | 583 | 619 |
| t0 known count | 17.6 | 10.1 | 62.4 | 102.5 |

- The overriding discriminator is **whether the true terms are candidates at all** (pool_recall rho
  vs delivered-f = **+0.530 LK / +0.515 PK**, the strongest of any axis; wrong proteins are 25% / 49%
  unreachable). Right proteins have near-complete pools; wrong proteins have terms missing.
- **Prior knowledge inverts between cells** (rho of t0-known-count vs delivered-f = **+0.305 LK** but
  **-0.278 PK**). In LK, more prior annotation = more context = better. In PK, more prior knowledge =
  WORSE, because the board's `-known` mask strips the easy known terms and leaves the deep, sparse,
  novel BP tail; the more richly a PK protein is already annotated, the harder the residual novelty.
  PK-BPO is a "novel-tail-on-already-rich-proteins" problem.

---

## The axes that actually carry a gradient (everything else is flat)

1. **Pool recall / reachability (dominant, both cells, rho ~+0.52).** The failure is a retrieval
   problem: delivered-f is governed by whether the true terms are in the pool.
2. **t0 known-count (rho +0.31 LK / -0.28 PK) -- and it INVERTS.** The single most surprising axis:
   prior knowledge helps LK and hurts PK (in the `-known` frame).
3. **Neighbour distance (LK only, rho -0.32; PK flat -0.02).** Homology twilight-zone hurts LK; PK
   proteins all have close homologs so it does not discriminate.

**Flat (real findings):** protein LENGTH (both, confirms PKW.3), TAXON (both -- loss is uniform
across model organisms, not concentrated in sparse taxa), STRING NETWORK DEGREE (flat for PK rho
-0.06; confounded-positive for LK; and 0% of failing proteins lack annotated partners), missed-term
IA band and DAG depth (both cells: ~77% of missed IA-mass is ordinary IA 1-6, ~72-74% at DAG depth
3-6 -- the missing terms are not exotic deep leaves, they are middling-specificity process terms).

## Recall-vs-ordering, restated as what is attackable

- The **ordering** fraction (in-pool, mis-ranked): 48.6% of LK missed mass, 33.6% of PK. In principle
  ranking-fixable, but the campaign's calibration ceiling says the deployed ranker is near-optimal
  here, so the realistic yield is small.
- The **recall** fraction (not in pool): 51.4% LK, 66.4% PK. Only GENERATION reaches it, and the
  campaign already showed the reachable-but-outside tail is not separable from the false tail by any
  in-house sequence-KNN + co-occurrence channel. The network channel (STRING) was the orthogonal
  probe; see below.

## The network axis (STRING v12.0) -- the orthogonal probe comes back negative

Only mouse (10090) and rat (10116) STRING v12.0 interactomes were downloaded, so this axis is
measurable for the mouse+rat subset: **182 LK-BPO** and **1,589 PK-BPO** proteins (~35% of each
cell). Human (9606, the other large block) STRING was not fetched, so the axis is silent on human
targets. Clean channels only (experimental + coexpression, score >0); annotated-partner = a partner
carrying a t0 (GAF v225) BP annotation.

| feature | LK-BPO | PK-BPO |
|---------|-------:|-------:|
| n with STRING feature | 181 | 1,574 |
| median clean-channel degree | 719 | 681 |
| frac with ZERO annotated partners | **0.00** | **0.00** |
| Spearman degree vs delivered-f | +0.391 | **-0.064** |
| Spearman annotated-partner-count vs delivered-f | +0.395 | **-0.067** |
| Spearman degree vs headroom | -0.372 | +0.040 |
| Spearman degree vs pool-recall (confound check) | +0.278 | +0.120 |

Two findings:
1. **The annotated-partner constraint is saturated.** Not one failing protein in either cell lacks
   annotated network partners (median degree ~700, all with annotated neighbours). We do NOT fail
   because the protein is network-isolated or its partners are uncharacterized. This confirms CG.3's
   feasibility note that ">=1 annotated partner is near-satisfied by construction" -- and it means the
   network hypothesis for the missing BP tail cannot be rescued by "these proteins had no partners to
   propagate from". They had hundreds. (Consistent with CG.3's negative result: the partners exist and
   are annotated, but their annotations do not propose the missing tail at a separable precision.)
2. **Network degree does NOT explain PK-BP failure** (flat, rho -0.064). For LK the positive
   degree/delivered-f correlation is a confound with pool recall and study depth (degree-vs-pool-recall
   rho +0.28), not an independent network signal. Being a hub does not help; being peripheral does not
   hurt. The BP frontier is orthogonal to network topology, at least for the mouse+rat block.

## Synthesis

Our BP frontier is a **retrieval frontier, not a protein-hardness frontier**. The proteins we fail on
are unremarkable -- ordinary length, ordinary organisms, with close homologs -- and the terms we miss
are ordinary middling-IA process terms, not exotic leaves. What separates success from failure is
almost entirely whether the candidate pool already contains the true term. LK-BPO is the milder,
half-ranking-half-retrieval case (high pool recall, cross-aspect context, oracle 0.79); PK-BPO is a
two-thirds generation wall on the novel tail of already-well-annotated proteins, made harder, not
easier, by prior knowledge. The only structured, non-diffuse headroom is the misordering sliver on
rich-pool proteins, which the campaign has shown is capped by cafaeval's cross-protein calibration.
Everything genuinely large is diffuse recall failure. What remains attackable is therefore an
orthogonal GENERATOR that proposes the missing middling-IA process terms. The STRING network channel
was the orthogonal probe for that generator, and this characterization narrows where it can help: the
failing proteins are NOT network-isolated and never lack annotated partners (0% with zero annotated
partners; median degree ~700), and network degree is flat against PK-BP success. So a network
generator cannot win by reaching proteins that had no partners -- they all had partners. Any network
gain must come from the partners' annotations proposing the specific missing terms at a precision that
survives cafaeval's mass metric, and CG.3 already found that precision (~0.4-1.6% IA) below the bar.
The honest frontier: the missing BP tail is reachable and its proteins are well-connected and
well-studied, but the true terms are neither in our pool nor separable from the false tail by
sequence-KNN, co-occurrence, or network-partner signal. The remaining lever is a better GENERATOR of
candidate terms, not a better ranker and not a network-reachability fix.
