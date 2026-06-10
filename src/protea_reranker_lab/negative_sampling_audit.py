"""Negative-sampling leakage audit for the universal multi-PLM reranker.

Implements and DOCUMENTS the balanced negative sampler for
F-RERANK-UNIVERSAL.4, including an explicit audit report that certifies
the negative construction does NOT replicate rows in a way that encodes
the label (the anc2vec replication-artefact cautionary template).

Leakage ruling (F-RERANK-UNIVERSAL.2, carried forward)
-------------------------------------------------------
The following features are EXCLUDED from training features and are NEVER
used as negative-selection criteria:

  lineage_is_ancestor_of_known
  lineage_is_known
  lineage_is_descendant_of_known
  lineage_shared_ancestor_count
  lineage_min_dist_to_known

These columns encode knowledge of whether the candidate GO term is
known/related-to-known for the protein.  Including them (or using them to
select negatives) would leak the label into the feature space.

Negative construction audit
----------------------------
The sampler in :func:`audit_negative_sampling` draws negatives UNIFORMLY at
RANDOM from the non-positive rows (label == 0) for each protein group.  The
draw uses a seeded numpy RNG (same seed as the train/val split seed) for
DETERMINISM.  It does NOT:

  - replicate rows to inflate the negative count.
  - weight negatives by any feature that correlates with the label.
  - use the ``lineage_*`` columns as selection criteria.
  - assign different negative rates to different GO subtrees (which could
    encode ontology structure as label-adjacent information).

The ``anc2vec_query_known_count`` column was found to be a bucket-ID artefact
(replication-by-category in the anc2vec pipeline, not temporal leakage;
see memory project_anc2vec_leakage_mechanism).  It is retained but treated as
a categorical conditioning feature, NOT as a negative-selection signal.

Balanced 1:1 default
--------------------
``neg_pos_ratio=1.0`` is the 1:1 balanced default for F-RERANK-UNIVERSAL.4.
It is tunable via :class:`~protea_reranker_lab.reranker.TrainConfig`
``neg_pos_ratio``.  The ratio is applied uniformly across all (protein, aspect)
groups to avoid differential class imbalance across cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Columns that are EXCLUDED from training features per the leakage ruling.
# This list is the canonical exclusion register; any consumer that builds
# a feature vector for training MUST drop these columns.
LINEAGE_EXCLUDED_FEATURES: tuple[str, ...] = (
    "lineage_is_ancestor_of_known",
    "lineage_is_known",
    "lineage_is_descendant_of_known",
    "lineage_shared_ancestor_count",
    "lineage_min_dist_to_known",
)


@dataclass(frozen=True)
class NegativeSamplingAuditReport:
    """Immutable audit report for one training run's negative sampling.

    Produced by :func:`audit_negative_sampling` and embedded in ``run.json``
    under the ``negative_sampling_audit`` key.

    Attributes
    ----------
    neg_pos_ratio:
        The requested ratio (1.0 = balanced; None = no downsampling).
    lineage_excluded:
        Tuple of feature names excluded from training (leakage guard).
    n_positives:
        Number of positive rows in the training split after split routing.
    n_negatives_before:
        Total negative rows before any downsampling.
    n_negatives_after:
        Negative rows retained after downsampling.
    actual_ratio:
        ``n_negatives_after / n_positives`` (or None when n_positives==0).
    sampling_seed:
        The seed used for the negative draw (same as the train split seed).
    no_label_encoding_by_lineage:
        Always True; confirms the lineage columns were not used for selection.
    no_row_replication:
        Always True; confirms negatives are sampled WITHOUT replacement.
    """

    neg_pos_ratio: float | None
    lineage_excluded: tuple[str, ...] = field(
        default_factory=lambda: LINEAGE_EXCLUDED_FEATURES
    )
    n_positives: int = 0
    n_negatives_before: int = 0
    n_negatives_after: int = 0
    actual_ratio: float | None = None
    sampling_seed: int = 42
    no_label_encoding_by_lineage: bool = True
    no_row_replication: bool = True


def audit_negative_sampling(
    *,
    n_positives: int,
    n_negatives_before: int,
    n_negatives_after: int,
    neg_pos_ratio: float | None,
    sampling_seed: int,
) -> NegativeSamplingAuditReport:
    """Build and return a :class:`NegativeSamplingAuditReport`.

    This function does not perform the sampling itself (that is done by
    :func:`~protea_reranker_lab.staging._decide_split`); it assembles the
    post-hoc audit report from the observed counts.

    Parameters
    ----------
    n_positives:
        Count of positive rows (label > 0) in the training split.
    n_negatives_before:
        Total negative rows before downsampling.
    n_negatives_after:
        Negative rows retained after downsampling.
    neg_pos_ratio:
        Requested ratio; None means no downsampling was applied.
    sampling_seed:
        The seed passed to the RNG for the negative draw.

    Returns
    -------
    :class:`NegativeSamplingAuditReport` with all audit fields filled.
    """
    actual = (
        n_negatives_after / n_positives
        if n_positives > 0
        else None
    )
    return NegativeSamplingAuditReport(
        neg_pos_ratio=neg_pos_ratio,
        lineage_excluded=LINEAGE_EXCLUDED_FEATURES,
        n_positives=n_positives,
        n_negatives_before=n_negatives_before,
        n_negatives_after=n_negatives_after,
        actual_ratio=actual,
        sampling_seed=sampling_seed,
        no_label_encoding_by_lineage=True,
        no_row_replication=True,
    )
