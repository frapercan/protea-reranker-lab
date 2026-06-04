Evaluation metrics
==================

This page documents every evaluation metric used in the reranker lab,
explains why the internal Fmax and the cafaeval Fmax differ by roughly
a factor of 10, defines Information Accretion (IA) formally, and
provides a per-cell equivalence table so that readers can locate each
number in the thesis and in the code without ambiguity.

.. contents:: On this page
   :depth: 2
   :local:

.. _metrics-internal-fmax:

Internal Fmax (proxy metric)
----------------------------

**Computed by:** :func:`protea_reranker_lab.evaluate.fmax_per_protein_group`

**What it measures.** Protein-averaged Fmax on the flat candidate rows
that LightGBM sees during training and internal validation. Inputs are
row-aligned numpy arrays: one score per (protein, GO-term) candidate,
one binary label (1 = positive, 0 = negative), and a ``group_sizes``
vector that records how many rows belong to each protein. The function
sweeps 101 thresholds, computes per-protein precision and recall at
each threshold, averages across proteins with at least one positive,
and returns the best F1.

**Key properties.**

- No DAG propagation. The candidates were retrieved by KNN; ancestor
  terms that were not among the K neighbours are absent from the input
  entirely. Precision and recall are measured only on the retrieved set.
- Evaluated under heavy class imbalance. The training pipeline uses
  ``neg_pos_ratio=10``, so for every true (protein, term) pair there
  are 10 negatives sampled from the candidate list. The internal Fmax
  therefore measures how well the model separates the correct term from
  a pool dominated by plausible distractors.
- No CAFA normalisation. The ``norm="cafa"`` correction (which divides
  by the number of proteins making at least one prediction above
  threshold) is not applied.

**Why it is low (0.07 to 0.28 for champion cells).** BPO is the hardest
aspect: many GO-term candidates per protein, very few positives. Under
``neg_pos_ratio=10`` the positive signal is diluted by design. MFO and
CCO are easier (fewer candidates, denser positives) and reach 0.28
internally. These numbers are correct proxies for the model's
discriminative ability on specific terms; they are not leaderboard
numbers.

**Do not cite in published output.** The internal Fmax is a training
and diagnostics metric only. Cite cafaeval Fmax (with annotation)
for leaderboard comparisons, and wFmax or S_min for the
IA-aligned claim.

.. _metrics-cafaeval-fmax:

cafaeval Fmax (CAFA-protocol metric)
-------------------------------------

**Computed by:** ``cafaeval-protea`` fork, function ``cafa_eval(...)``,
called from the sweep scripts (e.g. ``phase3d_K5_prostt5_sweep.py``).

**Call signature used in sweeps:**

.. code-block:: python

   cafa_eval(
       obo_file  = "datasets/bench-v1-K5/go.obo",
       pred_dir  = ...,   # per-protein prediction files
       gt_file   = ...,   # ground-truth TSV (candidates only)
       ia        = None,  # NOT passed: unwieghted Fmax
       prop      = "fill",
       norm      = "cafa",
       no_orphans= True,
       max_terms = 500,
       th_step   = 0.001,
       n_cpu     = 1,
   )

**What it measures.** CAFA-protocol Fmax: for each threshold, precision
and recall are computed after propagating every predicted term to all
ancestors via ``prop="fill"`` (fill-propagation: ancestors of a
predicted term are automatically predicted at the same score). The
threshold is swept and the best F1 is returned, averaged across proteins
with ``norm="cafa"``.

**Why it is 6 to 10 times higher than the internal Fmax.** Three
compounding effects:

1. **Propagation.** Fill-propagation adds all ancestors of every
   predicted term to the prediction set at no extra cost. The GO DAG
   is broad (MFO root GO:0003674 is an ancestor of virtually every MFO
   term), so even a single correct specific prediction drags many
   generic ancestors into the prediction. Precision in the denominator
   grows slowly while recall grows fast, pushing Fmax up.

2. **Restricted ground truth.** The ``gt_file`` passed to cafaeval
   contains only the positive terms that the KNN stage actually
   retrieved (``labels_p > 0`` in the candidate list). GO terms that
   are true annotations for a protein but were never retrieved by KNN
   are absent from the ground truth. The model therefore cannot be
   penalised for missing them: those false negatives are hidden. The
   cafaeval Fmax is conditional on the KNN retrieval and is optimistic
   relative to a full-pipeline CAFA submission.

3. **CAFA normalisation.** ``norm="cafa"`` counts a protein as
   having made a prediction only when at least one term is predicted
   above threshold, and excludes proteins with no predictions from the
   denominator. This matches CAFA competition rules but inflates the
   effective precision for borderline thresholds.

**Champion values (bench-v1-K3-v226-lineage-prostt5, seed 42):**

.. list-table::
   :header-rows: 1
   :widths: 20 25 20 20 20

   * - Cell
     - Internal Fmax
     - cafaeval Fmax
     - KNN baseline
     - Reranker lift
   * - lk-mfo
     - 0.278
     - 0.725
     - 0.582
     - +0.143
   * - lk-bpo
     - 0.069
     - 0.671
     - 0.584
     - +0.087
   * - lk-cco
     - 0.109
     - 0.781
     - 0.705
     - +0.076
   * - pk-mfo
     - 0.099
     - 0.511
     - (baseline)
     - n/a
   * - pk-bpo
     - 0.025
     - 0.379
     - (baseline)
     - n/a
   * - pk-cco
     - 0.063
     - 0.507
     - (baseline)
     - n/a

The reranker lift column shows the true contribution of the model over
the KNN baseline, which is the publishable result from Chapter 6.

.. _metrics-ia:

Information Accretion (IA)
--------------------------

Information Accretion is the formal measure of how much
ontology-specific information a GO-term prediction adds beyond what
a prediction of its parent terms already implies. It was introduced
by Clark and Radivojac (2013) and is the basis of the CAFA weighted
metrics (wFmax, S_min).

**Formal definition.**

Given a set of reference proteins :math:`R` and a GO term :math:`t`
with direct parents :math:`\text{par}(t)`:

.. math::

   \text{IA}(t)
   = -\log_2 P(t \mid \text{par}(t))
   = -\log_2 \frac{
       |\{p \in R : t \in \text{annot}(p)\}|
     }{
       |\{p \in R : \forall s \in \text{par}(t),\, s \in \text{annot}(p)\}|
     }

where :math:`\text{annot}(p)` is the propagated (closed-under-ancestors)
annotation set of protein :math:`p`. The denominator counts proteins
annotated with ALL parents of :math:`t`; the numerator counts those
additionally annotated with :math:`t` itself. The ratio is the
conditional probability that a protein already satisfying the parents
is also annotated to the child term. A highly specific term with very
few annotated proteins yields high IA (high surprise); a very generic
term whose annotation is almost guaranteed by its parents yields IA
close to 0.

Root terms (GO:0008150 Biological process, GO:0005575 Cellular
component, GO:0003674 Molecular function) have no parents; by
convention their IA is defined to be 0.

**IA file used in this lab:**

``datasets/ia/IA-swissprot-exp-v227.txt``

Format: two-column tab-separated (``GO_term<TAB>IA_bits``), no header.
This is the exact format consumed by ``cafaeval``'s ``ia_parser``.

.. _metrics-ia-provenance:

IA provenance
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - Corpus
     - SwissProt (reviewed UniProtKB) experimental GOA annotations
   * - GOA release
     - v227 (September 2025), PROTEA annotation_set ``5e84d6c5-...``
   * - Evidence codes
     - EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP, IC, TAS (13 codes)
   * - Excluded codes
     - IEA, all computational (ISS, ISO, ISA, ISM, IGC, RCA), all phylogenetic
       (IBA, IBD, IKR, IRD), NAS, ND, and every ``NOT`` qualifier
   * - Ontology (OBO)
     - ``datasets/bench-v1-K5/go.obo`` (``releases/2026-01-23``)
   * - OBO edges for propagation
     - ``is_a`` and ``part_of`` only, no cross-aspect edges
   * - Tooling
     - ``democafa`` commit ``742814eb`` (github.com/anphan0828/democafa_package)
   * - Entry point
     - ``python -m democafa.datacollection.propagate_and_ia``
   * - IA rows written
     - 38,739 (full OBO term universe across P, C, F)
   * - Rows with nonzero IA
     - 28,391
   * - IA range
     - 0 to 15.9462 bits
   * - Mean IA
     - 2.773 bits

Per-aspect statistics over terms present in the corpus:

.. list-table::
   :header-rows: 1
   :widths: 10 12 10 10 12 12 12

   * - Aspect
     - Terms
     - Min (bits)
     - Max (bits)
     - Mean (bits)
     - Median (bits)
     - Nonzero
   * - CCO (C)
     - 2,959
     - 0.000
     - 14.894
     - 3.532
     - 2.485
     - 2,577
   * - MFO (F)
     - 7,283
     - 0.000
     - 14.924
     - 3.763
     - 3.442
     - 6,666
   * - BPO (P)
     - 19,244
     - 0.000
     - 13.828
     - 1.778
     - 0.485
     - 13,364

Validation spot checks confirm correct behaviour: shallow MF terms have
low IA (GO:0003824 catalytic activity 1.499 bits; GO:0016787 hydrolase
activity 1.750 bits), locational terms are intermediate (GO:0005634
nucleus 0.954 bits; GO:0005737 cytoplasm 0.347 bits), and singleton
terms approach the per-aspect maximum.

.. _metrics-wfmax:

Weighted Fmax (wFmax) and S_min
--------------------------------

**wFmax** is the IA-weighted analogue of cafaeval Fmax. Precision and
recall at each threshold are replaced by their IA-weighted counterparts:

.. math::

   \text{ru}(t, \tau) = \frac{\sum_{t : \hat{s}(t) < \tau} \text{IA}(t)}{
     \sum_{t \in A} \text{IA}(t)}

   \text{mi}(t, \tau) = \frac{\sum_{t : \hat{s}(t) \geq \tau,\, t \notin A} \text{IA}(t)}{
     \text{normaliser}}

where :math:`A` is the ground-truth annotation set, :math:`\hat{s}(t)`
is the predicted score for term :math:`t`, ``ru`` is remaining
uncertainty (weighted false negative rate), and ``mi`` is misinformation
(weighted false positive rate). The best F1-like combination over
thresholds is wFmax; the minimum Euclidean distance ``ru`` + ``mi``
is S_min. Both are available from ``cafaeval`` when ``ia=`` is passed.

**Why wFmax / S_min are the honest metrics for this hypothesis.** The
IA-alignment hypothesis claims that training the reranker to weight
specific (high-IA) terms more heavily than generic (low-IA) ones will
improve the scores that the IA weighting actually measures. A model that
improves wFmax over the KNN baseline has genuinely lifted the signal in
the informative region, where propagation cannot inflate the number.
The unweighted cafaeval Fmax can stay flat or decrease slightly: that
happens when the model shifts probability mass from generic to specific
terms (lower precision on easy terms, higher on hard ones), which is the
desired behaviour.

.. _metrics-recovery-ceiling:

Recovery ceiling
----------------

The cafaeval Fmax is optimistic because the ground truth is restricted
to the candidates that the KNN stage retrieved. The **recovery ceiling**
quantifies this optimism: it is the fraction of a protein's true
positive GO terms (propagated, for the aspect under evaluation) that
appear in the KNN candidate list.

Formally, for protein :math:`p`, aspect :math:`a`, and KNN candidate
set :math:`C_p`:

.. math::

   \text{ceiling}(p, a) =
   \frac{|\text{annot}(p, a) \cap C_p|}{|\text{annot}(p, a)|}

A ceiling of 0.80 means that 20 percent of the protein's true annotations
could not possibly be recovered, regardless of how good the reranker is,
because the KNN stage never produced them as candidates. The cafaeval
Fmax computed in the lab is therefore an upper bound on what a full
pipeline would achieve.

**Decomposition.** Pipeline performance decomposes into two independent
stages:

1. **KNN recall** (retrieval ceiling): the fraction of true annotations
   that appear in the candidate list.
2. **Reranker precision at threshold** (ranking quality): given the
   candidates, how accurately the model ranks positives above negatives.

Chapter 6 separates these two quantities so that the reranker's
contribution (stage 2) is not conflated with the KNN's limitations
(stage 1). The per-cell delta over the KNN baseline (columns in the
table above) measures stage 2 in isolation.

.. _metrics-equivalence-table:

Per-cell equivalence table
--------------------------

The following table places the four metric families side by side for
the pilot cell set (``bench-v1-K3-v226-lineage-prostt5``, seed 42,
champion reranker). The wFmax and S_min columns are placeholders
pending the completion of the IA-aligned re-evaluation run
(executor slice F-LAFA-IA.0). Once those runs complete, the results
will be placed under ``runs/`` and this table will be updated.

.. list-table::
   :header-rows: 1
   :widths: 12 18 18 18 18 16

   * - Cell
     - Internal Fmax
     - cafaeval Fmax
     - wFmax (IA-weighted)
     - S_min (IA-weighted)
     - KNN cafaeval Fmax
   * - lk-mfo
     - 0.278
     - 0.725
     - *TBD*
     - *TBD*
     - 0.582
   * - lk-bpo
     - 0.069
     - 0.671
     - *TBD*
     - *TBD*
     - 0.584
   * - lk-cco
     - 0.109
     - 0.781
     - *TBD*
     - *TBD*
     - 0.705
   * - nk-mfo
     - n/a
     - 0.7065 (mean 3 seeds)
     - *TBD*
     - *TBD*
     - 0.6447
   * - nk-bpo
     - n/a
     - 0.5596 (mean 3 seeds)
     - *TBD*
     - *TBD*
     - 0.5333
   * - nk-cco
     - n/a
     - 0.7774 (mean 3 seeds)
     - *TBD*
     - *TBD*
     - 0.7000
   * - pk-mfo
     - 0.099
     - 0.511
     - *TBD*
     - *TBD*
     - 0.4831
   * - pk-bpo
     - 0.025
     - 0.379
     - *TBD*
     - *TBD*
     - 0.4031
   * - pk-cco
     - 0.063
     - 0.507
     - *TBD*
     - *TBD*
     - 0.6009

Internal Fmax is computed without propagation on the raw candidate rows.
The cafaeval Fmax uses ``prop="fill"``, ``norm="cafa"``, no IA weighting.
wFmax and S_min use the same cafaeval call with
``ia="datasets/ia/IA-swissprot-exp-v227.txt"`` (IA provenance: see
:ref:`metrics-ia-provenance`). Cells labelled *TBD* will be filled once
the re-evaluation results are written to ``runs/``.

.. _metrics-references:

References
----------

- Clark, W.T. and Radivojac, P. (2013). Information-theoretic evaluation
  of predicted ontological annotations. *Bioinformatics*, 29(13),
  i53-i61. https://doi.org/10.1093/bioinformatics/btt228

- Jiang, Y. et al. (2016). An expanded evaluation of protein function
  prediction methods with refined community standards. *Nucleic Acids
  Research*, 44(15), e123. (CAFA 2 normalisation conventions)

- Radivojac, P. et al. (2013). A large-scale evaluation of computational
  protein function prediction. *Nature Methods*, 10, 221-227. (CAFA 1)

- democafa package: https://github.com/anphan0828/democafa_package
  (commit ``742814eb``)

- IA artefact and full reproduction steps: ``datasets/ia/provenance.md``
  and ``datasets/ia/build_corpus.sql``

- cafaeval-protea fork: ``~/Thesis2/repositories/cafaeval-protea``
  (function signature in ``src/cafaeval/evaluate.py::cafa_eval``)
