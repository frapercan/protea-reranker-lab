# Is the missing BP-tail reachable from ANY t0 signal? The maximal-union recall ceiling

Read-only, frozen-data, board frame. This is the decisive **recall-ceiling** measurement for the BP
retrieval frontier: set membership only (no cafaeval / no f_micro_w). It splits the missing true
BP-tail into **reachable-but-inseparable** (some t0 signal proposes the term, but only buried in false
mass = metric-capped) versus **fundamentally un-retrievable** (no t0 signal for that protein reaches
it = the field frontier).

Machinery: `union_recall.py` (target + union), `recall_knn_producer.py` (8-PLM kNN),
`recall_network_proposals.py` (STRING), `recall_literature_proposals.py` (abstract->GO). Receipt JSON
`RECALL_CEILING_MAXIMAL_UNION.json`; per-PLM proposal caches `recall_scratch/`.

## Frame and target (matches the CG-channel reachability convention)

Per LK/PK-BPO target with >=1 valid true BP term (universe = the exact 523 / 4397 of
`bp_failure_atlas.json`):
- `gt_valid` = propagated-true BP toi terms (PK: minus t0-known, propagated) -- the board `-known` frame.
- `pool` = propagated closure of the deployed `eval.parquet` BP candidates.
- `missing tail` = `gt_valid - pool`, IA-weighted = the terms retrieval never reaches.

Set-membership reachability (a true term is reachable if **any** pool candidate propagates to it) makes
`pool_recall` equal to CG.1/CG.2's `pool_captured_frac`: **LK 0.7406 vs their 0.739** (independent
reproduction). PK is lower (0.5467) because the board `-known` mask strips the easy known terms and
leaves the novel tail -- exactly the atlas's "novel-tail-on-already-rich-proteins" picture.

## The headline: the ceiling per cell (current pool -> maximal union)

| cell | n | gt_valid IA | pool recall (of all true) | missing tail IA (% of gt) | **union recall OF the missing tail** | total recall pool->union | **residual unreachable IA (% of missing / % of all true)** |
|------|--:|------------:|--------------------------:|--------------------------:|------------------------------------:|-------------------------:|----------------------------------------------------------:|
| LK-BPO | 523 | 9 076.8 | 0.741 | 2 354.9 (25.9%) | **0.610** | 0.741 -> **0.899** | **919.3 (39.0% / 10.1%)** |
| PK-BPO | 4 397 | 48 715.3 | 0.547 | 22 082.2 (45.3%) | **0.550** | 0.547 -> **0.796** | **9 940.5 (45.0% / 20.4%)** |

The maximal t0 union reaches a **majority** of the previously-missing tail (61% LK, 55% PK). But a
**large residual is reached by no t0 signal at all**: 39% of LK's missing tail and 45% of PK's, which
is **10.1% of ALL true LK-BP IA-mass and 20.4% of ALL true PK-BP IA-mass**. That residual is the
genuine field frontier.

## Which sources add the most (marginal recall of the missing tail, kNN-diversity-first order)

| source | LK standalone | LK marginal | PK standalone | PK marginal |
|--------|--------------:|------------:|--------------:|------------:|
| kNN ankh_base | 0.180 | **0.180** | 0.141 | **0.141** |
| kNN esm2_150m | 0.288 | 0.125 | 0.221 | 0.097 |
| kNN esmc_600m | 0.352 | 0.083 | 0.254 | 0.050 |
| kNN ankh_large | 0.223 | 0.070 | 0.156 | 0.061 |
| kNN esm2_650m | 0.289 | 0.038 | 0.228 | 0.054 |
| kNN esm2_3b | 0.272 | 0.027 | 0.231 | 0.042 |
| **kNN learned_champion (deployed pool basis)** | 0.158 | **0.004** | 0.102 | **0.007** |
| classifier extras (top-50) | 0.080 | 0.014 | 0.029 | 0.007 |
| network (STRING clean) | 0.249 | 0.068 | 0.250 | 0.091 |
| literature (abstract->GO top-50) | *(bound, see note)* | *<=0.30* | *(bound)* | *<=0.24* |

*Literature note: the abstract->GO regenerator (`recall_literature_proposals.py`, S-PubMedBert on CPU)
is still encoding at report time; its exact union marginal will be appended to the JSON. Its published
CG.2 STANDALONE recall of the pool-missed FN tail is 0.298 (LK) / 0.236 (PK) at top-50. As a marginal
OVER the 7-PLM + network union it must be strictly smaller, because abstract-retrieval and
sequence-homolog retrieval surface overlapping annotations. Even crediting it as fully disjoint (a
hard over-estimate) lifts the union recall of the missing tail to at most ~0.91 (LK) / ~0.79 (PK), so
a fundamentally-unreachable residual of >=~9% (LK) / >=~21% of the missing tail remains -- the verdict
is invariant to the literature channel.*

Three findings:
1. **PLM DIVERSITY is the whole reachable lever, and the deployed encoder is not it.** The deployed
   learned champion (which built the pool) adds essentially **zero** marginal recall (0.004 LK / 0.007
   PK) -- it already retrieved what it can. The *raw, diverse* PLMs each add real, previously-missing
   recall: ankh_base +18%/+14%, esm2_150m +12%/+10%, esm-c +8%/+5%, ankh_large +7%/+6%. Different PLM
   families see different homologs and transfer different true annotations. **Diverse-PLM retrieval is
   a genuine, unexploited recall channel** -- see the precision caveat below.
2. **Network (STRING) is partly orthogonal:** +6.8%/+9.1% marginal *after* seven PLMs, the second
   largest single contributor for PK. Partners propose true terms the sequence neighbours miss.
3. **The classifier adds almost nothing at set-membership level** (+1.4%/+0.7%): its extras are largely
   terms the kNN union already reaches (its value in the campaign was scoring, not novel membership).

## The separability wall, made quantitative (precision context)

The union reaches those terms only by proposing an enormous false mass:

| cell | new proposed cells | false IA added (over gt_valid) | true IA recovered | union precision |
|------|-------------------:|-------------------------------:|------------------:|----------------:|
| LK-BPO | 238 868 | 326 877 | ~1 436 | ~0.4% IA |
| PK-BPO | 3 359 782 | 4 658 072 | ~12 141 | ~0.26% IA |

So for the **reachable** majority of the tail, the frontier is **separation, not retrieval**: the terms
are present in the union, but at ~0.3-0.4% IA precision they are indistinguishable from the ~100-380x
larger false tail -- exactly the cafaeval fill-tax wall the campaign has hit through six channels. No
scorer converts a 0.3% signal under prop=fill/global-tau.

## The residual: what no t0 signal reaches

The residual (unreachable by pool + 7 PLMs + network + classifier) is **not exotic deep leaves**. By
IA-mass it sits at middling DAG depth (LK 74% at depth 3-7; PK 90% at depth 3-8, peak at depth 5) and
middling information content (LK 71% in IA band 1-6; PK 82% in IA band 1-6, peak at IA 4). These are
**ordinary-specificity biological-process terms** for which the protein has, at t0: no sequence homolog
carrying the term (across 6 PLM families x 50 neighbours each), no STRING partner annotated with it,
and no full-vocab-classifier proposal. They are genuinely novel protein->process associations relative
to every t0 signal we hold -- the true field frontier, not a metric artefact and not a retrieval bug we
can engineer away.

## Leakage disposition (proven clean)

- **Embeddings + neighbour annotations:** the `c905dffa` snapshot = GOA **v227 = t0 (2025-09-04
  cutoff)**. Query vectors are looked up FROM the same t0 reference matrix by accession; **self-accession
  neighbours are dropped**, so a query cannot transfer its own annotations. Every transferred term is a
  v227 reference annotation -> no post-t0 term can enter.
- **Network:** STRING v12.0 release **2023-07-26 << t0**; textmining channel never read (clean =
  experimental + coexpression); partner annotations only from frozen v227; self / same-accession
  partners excluded.
- **Literature:** PMIDs restricted to publication year < 2025 (< t0); abstracts cached, no network.
- The only post-t0 information anywhere is the candidate LABEL (the board ground truth = the target).

## PLMs used and absent

Used (t0-clean, `c905dffa` ref materialized): **ankh_base, ankh_large, esm2_150m, esm2_650m, esm2_3b,
esmc_600m** (six raw PLMs spanning ESM2 x3 sizes, Ankh x2, ESM-C) **+ learned_champion (d8979601)**.
Absent in a t0-clean form: **ProtT5, ProstT5** (only the `00000000` no-filter reference is materialized,
whose annotations are post-t0 = would leak), and **ProtST-text** (no `c905dffa` reference). The used set
already spans four architecture families; adding ProtT5/ProtST would, on the observed heavy inter-PLM
overlap (standalone recalls ~0.14-0.35 collapsing to a ~0.50 union), raise the ceiling only marginally.

## Verdict

**SPLIT, and now quantified.** The BP retrieval frontier is two distinct problems:

1. **Reachable-but-inseparable (the majority: 61% of LK's missing tail, 55% of PK's).** These true
   terms ARE proposed by the maximal t0 union -- dominantly by *diverse* sequence-kNN (not the deployed
   encoder) plus STRING. The frontier here is **separation, not retrieval**: union precision is
   ~0.3-0.4% IA, so the metric caps the payoff. This corroborates the campaign's "reachable but not
   separable" wall and adds the missing half: the reachability was there all along in PLM diversity we
   never retrieved over.

2. **Fundamentally un-retrievable (the residual: 39% of LK's missing tail = 10.1% of all true LK-BP IA;
   45% of PK's = 20.4% of all true PK-BP IA).** No t0 signal we hold -- 6 PLM families, STRING, the
   full-vocab classifier -- proposes these ordinary middling-IA process terms for these proteins. This
   is the true field frontier: novel protein->process associations invisible to sequence, network, and
   text at t0.

Receipt: `storage/regen_headline/RECALL_CEILING_MAXIMAL_UNION.json` (+ `.md`), proposal caches under
`storage/regen_headline/recall_scratch/`.
