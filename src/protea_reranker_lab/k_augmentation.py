"""Seeded, bounded K-augmentation for the universal multi-PLM reranker.

K-augmentation draws training candidates from a SEEDED, bounded K
distribution across the K{3,5,10} pooled sources.  A deterministic K
policy governs inference (never an unseeded stream).  Both the seed, the
bounds, and the inference policy are captured in :class:`ExperimentSpec`
(via :class:`TrainingSpec`) so changes to any of them invalidate the
cached spec hash.

Design constraints (F-RERANK-UNIVERSAL.4)
------------------------------------------
- numpy only; NEVER torch GPU or pgvector.
- Two runs with the SAME seed MUST produce a byte-identical candidate
  selection (the byte-identical-twice test gate verifies this).
- ``k_aug_bounds=(k_min, k_max)`` draws K uniformly at random from the
  closed integer interval [k_min, k_max] PER SOURCE GROUP (one draw per
  distinct protein in the pooled batch).
- ``k_inference_policy="fixed"`` uses ALL pool sources unchanged (no
  stochastic draw); ``"adaptive"`` selects K=max available for each
  protein (equivalent to the largest-K superset, deterministic).
- ``k_aug_bounds=None`` means NO augmentation: pass all sources through.

Leakage audit
-------------
K-augmentation operates on the already-leakage-audited candidate set
produced by :mod:`multi_source` (``lineage_*`` columns EXCLUDED per the
F-RERANK-UNIVERSAL.2 ruling).  The K selection does NOT re-introduce
lineage information: it only decides how many KNN neighbors to include
per protein, which is a retrieval-depth parameter, not a leakage source.
The anc2vec replication artefact (the cautionary template) would arise
from replicating rows in a way that encodes the label; K-augmentation
avoids that by sampling WITHOUT REPLACEMENT from the distinct K values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical K values in the pooled multi-source dataset family.
CANONICAL_K_VALUES: tuple[int, ...] = (3, 5, 10)

_VALID_INFERENCE_POLICIES = ("fixed", "adaptive")


@dataclass(frozen=True)
class KAugPolicy:
    """Seeded, bounded K-augmentation policy.

    Attributes
    ----------
    seed:
        RNG seed for the K draw.  Identical seeds produce byte-identical
        draws (determinism gate).
    k_min, k_max:
        Inclusive bounds for the draw.  Both must be members of
        :data:`CANONICAL_K_VALUES` (3, 5, or 10).
    inference_policy:
        "fixed" (default) or "adaptive".  "fixed" = use all pool sources
        unmodified at inference.  "adaptive" = use the largest K available
        per protein.
    """

    seed: int
    k_min: int
    k_max: int
    inference_policy: str = "fixed"

    def __post_init__(self) -> None:
        if self.k_min > self.k_max:
            raise ValueError(
                f"k_min ({self.k_min}) must be <= k_max ({self.k_max})"
            )
        for val in (self.k_min, self.k_max):
            if val not in CANONICAL_K_VALUES:
                raise ValueError(
                    f"k_aug bounds must be members of {CANONICAL_K_VALUES}; "
                    f"got {val}"
                )
        if self.inference_policy not in _VALID_INFERENCE_POLICIES:
            raise ValueError(
                f"inference_policy must be one of {_VALID_INFERENCE_POLICIES}; "
                f"got {self.inference_policy!r}"
            )

    @property
    def eligible_k_values(self) -> list[int]:
        """Sorted list of K values in [k_min, k_max]."""
        return sorted(k for k in CANONICAL_K_VALUES if self.k_min <= k <= self.k_max)


def sample_k_per_protein(
    proteins: np.ndarray,
    policy: KAugPolicy,
) -> np.ndarray:
    """Draw a K value for each UNIQUE protein using the seeded RNG.

    Returns a 1-D int32 array aligned to ``np.unique(proteins)`` (sorted
    order, same as numpy unique's default).  Two calls with the same
    ``proteins`` array and the same ``policy.seed`` return a BYTE-IDENTICAL
    result.

    Parameters
    ----------
    proteins:
        Row-aligned protein accession array (string dtype).
    policy:
        Augmentation policy specifying seed and bounds.

    Returns
    -------
    Array of shape ``(n_unique_proteins,)`` with K values drawn uniformly
    at random from ``policy.eligible_k_values``.
    """
    unique_proteins = np.unique(proteins)
    eligible = np.array(policy.eligible_k_values, dtype=np.int32)
    if len(eligible) == 0:
        return np.full(len(unique_proteins), CANONICAL_K_VALUES[0], dtype=np.int32)
    rng = np.random.default_rng(policy.seed)
    indices = rng.integers(0, len(eligible), size=len(unique_proteins))
    return eligible[indices]


def build_protein_k_map(
    proteins: np.ndarray,
    policy: KAugPolicy,
) -> dict[str, int]:
    """Return a ``{protein: k_value}`` mapping for the augmented K draw.

    This is the deterministic map used to filter the multi-source pool at
    staging time: for each protein, only rows from sources whose
    ``k_context <= k_map[protein]`` are retained.  This implements a
    K-superset draw: K=10 includes all K{3,5,10} neighbors for that protein,
    K=5 includes K{3,5}, K=3 includes K{3} only.
    """
    unique_proteins = np.unique(proteins)
    k_draws = sample_k_per_protein(proteins, policy)
    return {str(p): int(k) for p, k in zip(unique_proteins, k_draws)}


def filter_rows_by_k_map(
    proteins: np.ndarray,
    k_contexts: np.ndarray,
    k_map: dict[str, int],
) -> np.ndarray:
    """Return a boolean mask keeping only rows whose k_context <= k_map[protein].

    Rows for proteins absent from ``k_map`` are ALWAYS kept (conservative
    fallback for proteins not seen during the pool construction).

    Parameters
    ----------
    proteins:
        Row-aligned protein accession array.
    k_contexts:
        Row-aligned K context values (int).
    k_map:
        Per-protein K ceiling from :func:`build_protein_k_map`.

    Returns
    -------
    Boolean mask of shape ``(len(proteins),)``.
    """
    mask = np.ones(len(proteins), dtype=bool)
    for i, (p, k) in enumerate(zip(proteins, k_contexts)):
        ceiling = k_map.get(str(p))
        if ceiling is not None and int(k) > ceiling:
            mask[i] = False
    return mask


def inference_k_sources(
    k_values: list[int],
    policy: KAugPolicy,
) -> list[int]:
    """Return the K values to include at inference time per the policy.

    ``"fixed"`` returns all K values unchanged (use the full pool).
    ``"adaptive"`` returns only the largest K (equivalent to the K-superset
    of all available neighbours).

    Parameters
    ----------
    k_values:
        Distinct K values present in the inference pool.
    policy:
        Augmentation policy.

    Returns
    -------
    Sorted list of K values to include at inference.
    """
    if not k_values:
        return []
    if policy.inference_policy == "adaptive":
        return [max(k_values)]
    return sorted(k_values)
