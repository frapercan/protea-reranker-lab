"""Sparse Distributed Representation (SDR) primitives for the dense<->sparse axis.

This module implements the cheapest decisive test of the ``sparse.pdf`` hypothesis
(Appendix A.2, readout-1): *an interpretable set-overlap in a sparse space can be at
least as good a proxy for biological similarity as a distance in a dense space.*

The pieces are deliberately small and pure so they can be unit-tested and later
reused as the ``metric="tanimoto"`` branch of ``protea-method``'s ``search_knn`` and
as an ``EvidenceScorer`` in the stacked meta-reranker (ADR-D43):

* :func:`kwta_active_set` / :func:`kwta_binarise` turn a dense embedding into a binary
  Sparse Distributed Representation by keeping the top-k entries by magnitude (k-WTA).
* :func:`tanimoto_dense` computes the Tanimoto set-overlap similarity between SDRs.
* :func:`cosine_dense` is the dense baseline arm over the raw embeddings.
* :class:`GoDag` parses a ``go-basic.obo`` snapshot and exposes ancestor closures.
* :func:`information_content` derives Resnik information content from a corpus of
  (propagated) annotation sets.
* :func:`resnik_pairwise` / :func:`lin_pairwise` give GO semantic similarity over the
  DAG (the biological-similarity ground truth for the correlation readout).

Everything is leakage-clean by construction: it consumes only a t0 reference pool and
the matching t0 ontology snapshot; nothing here ever reads post-cutoff (t1) data.
"""

from __future__ import annotations

import gzip
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Aspect -> the GO root term whose IC must be pinned to zero (a protein annotated
# only at a root carries no information; Resnik IC of the root is 0 by definition).
GO_ROOTS = {
    "P": "GO:0008150",  # biological_process
    "F": "GO:0003674",  # molecular_function
    "C": "GO:0005575",  # cellular_component
}


# ---------------------------------------------------------------------------
# k-WTA binarisation + Tanimoto (the SPARSE arm)
# ---------------------------------------------------------------------------
def kwta_active_set(vec: np.ndarray, k: int) -> np.ndarray:
    """Return the sorted indices of the ``k`` largest-magnitude entries of ``vec``.

    The active set is the support of the k-WTA Sparse Distributed Representation:
    the indices that are "on" once the dense vector is binarised. Ties beyond ``k``
    are broken by index order (numpy ``argpartition`` semantics), which is stable
    enough for a correlation readout.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= vec.shape[-1]:
        return np.arange(vec.shape[-1], dtype=np.int64)
    idx = np.argpartition(np.abs(vec), -k)[-k:]
    idx.sort()
    return idx.astype(np.int64)


def kwta_binarise(emb: np.ndarray, k: int) -> np.ndarray:
    """Binarise a ``(n, d)`` embedding matrix into a ``(n, d)`` uint8 k-WTA SDR.

    For each row the ``k`` largest-magnitude coordinates are set to 1, the rest to 0.
    The result is a dense bitset; ``tanimoto_dense`` consumes it directly.
    """
    if emb.ndim != 2:
        raise ValueError("emb must be a 2-D (n, d) matrix")
    n, d = emb.shape
    if k >= d:
        return np.ones((n, d), dtype=np.uint8)
    # top-k magnitude indices per row, vectorised
    part = np.argpartition(np.abs(emb), d - k, axis=1)[:, d - k:]
    out = np.zeros((n, d), dtype=np.uint8)
    rows = np.repeat(np.arange(n), k)
    out[rows, part.reshape(-1)] = 1
    return out


def tanimoto_dense(sdr: np.ndarray, k: int) -> np.ndarray:
    """Pairwise Tanimoto similarity matrix for a ``(n, d)`` k-WTA bitset.

    Each row has exactly ``k`` active bits, so for two rows A, B with intersection
    ``i = |A & B|`` the Tanimoto coefficient is ``i / (2k - i)`` (sparse.pdf App. B).
    Returns an ``(n, n)`` float matrix; the diagonal is 1.
    """
    sdr_f = sdr.astype(np.float32)
    inter = sdr_f @ sdr_f.T  # |A intersect B| for every pair
    denom = (2 * k) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, inter / denom, 0.0)
    return out.astype(np.float32)


def cosine_dense(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for a ``(n, d)`` dense matrix (the BASELINE)."""
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    unit = emb / norm
    return (unit @ unit.T).astype(np.float32)


# ---------------------------------------------------------------------------
# GO DAG (the t0 ontology snapshot) + information content (the SEMANTIC arm)
# ---------------------------------------------------------------------------
class GoDag:
    """A read-only is_a / part_of GO DAG parsed from a ``go-basic.obo`` snapshot.

    Only the structure needed for ancestor closures and Resnik/Lin similarity is
    kept: term -> direct parents (is_a + part_of), term -> namespace (aspect), the
    obsolete set, and the alt_id -> primary id remap. The DAG is the t0 ontology;
    pass the OBO release matching the reference-pool cutoff to stay leakage-clean.
    """

    _NS_TO_ASPECT = {
        "biological_process": "P",
        "molecular_function": "F",
        "cellular_component": "C",
    }

    def __init__(
        self,
        parents: dict[str, frozenset[str]],
        aspect: dict[str, str],
        alt_ids: dict[str, str],
    ) -> None:
        self.parents = parents
        self.aspect = aspect
        self.alt_ids = alt_ids
        self._ancestor_cache: dict[str, frozenset[str]] = {}

    @classmethod
    def from_obo(cls, path: str | Path) -> "GoDag":
        """Parse a ``go-basic.obo`` (optionally gzipped) into a :class:`GoDag`."""
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        parents: dict[str, set[str]] = defaultdict(set)
        aspect: dict[str, str] = {}
        alt_ids: dict[str, str] = {}

        cur_id: str | None = None
        cur_ns: str | None = None
        obsolete = False
        in_term = False

        def _flush() -> None:
            if cur_id and not obsolete:
                if cur_ns in cls._NS_TO_ASPECT:
                    aspect[cur_id] = cls._NS_TO_ASPECT[cur_ns]
                parents.setdefault(cur_id, set())

        with opener(path, "rt") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line == "[Term]":
                    _flush()
                    cur_id, cur_ns, obsolete, in_term = None, None, False, True
                    continue
                if line.startswith("["):  # [Typedef] etc. ends the term block
                    _flush()
                    in_term = False
                    continue
                if not in_term or not line:
                    continue
                if line.startswith("id: GO:"):
                    cur_id = line[4:].strip()
                elif line.startswith("namespace: "):
                    cur_ns = line[len("namespace: "):].strip()
                elif line.startswith("is_obsolete: true"):
                    obsolete = True
                elif line.startswith("is_a: GO:") and cur_id:
                    parents[cur_id].add(line[len("is_a: "):].split("!")[0].strip())
                elif line.startswith("relationship: part_of GO:") and cur_id:
                    parents[cur_id].add(
                        line[len("relationship: part_of "):].split("!")[0].strip()
                    )
                elif line.startswith("alt_id: GO:") and cur_id:
                    alt_ids[line[len("alt_id: "):].strip()] = cur_id
            _flush()

        return cls(
            parents={t: frozenset(p) for t, p in parents.items()},
            aspect=aspect,
            alt_ids=alt_ids,
        )

    def primary(self, go_id: str) -> str:
        """Map an alt_id to its primary id (identity if already primary)."""
        return self.alt_ids.get(go_id, go_id)

    def ancestors(self, go_id: str) -> frozenset[str]:
        """All is_a / part_of ancestors of ``go_id`` INCLUDING ``go_id`` itself.

        Obsolete / unknown terms collapse to the empty set, so a stray label never
        crashes the closure.
        """
        go_id = self.primary(go_id)
        cached = self._ancestor_cache.get(go_id)
        if cached is not None:
            return cached
        if go_id not in self.parents:
            empty: frozenset[str] = frozenset()
            self._ancestor_cache[go_id] = empty
            return empty
        acc: set[str] = {go_id}
        stack = list(self.parents[go_id])
        while stack:
            p = stack.pop()
            if p in acc:
                continue
            acc.add(p)
            stack.extend(self.parents.get(p, ()))
        out = frozenset(acc)
        self._ancestor_cache[go_id] = out
        return out


def propagate(go_ids: Iterable[str], dag: GoDag) -> frozenset[str]:
    """True-Path-Rule closure of a protein's leaf annotations over the DAG."""
    acc: set[str] = set()
    for g in go_ids:
        acc |= dag.ancestors(g)
    return frozenset(acc)


def information_content(
    annotation_sets: Iterable[frozenset[str]],
    dag: GoDag,
) -> dict[str, float]:
    """Resnik information content per GO term from a corpus of PROPAGATED sets.

    ``IC(t) = -log( freq(t) / freq(root_of_t's_aspect) )``. Counting is done over
    already-propagated sets (each set is the ancestor closure of one protein), so a
    term's frequency includes every descendant annotation. Roots get IC 0.
    """
    counts: dict[str, int] = defaultdict(int)
    n_sets = 0
    for s in annotation_sets:
        n_sets += 1
        for t in s:
            counts[t] += 1
    if n_sets == 0:
        return {}

    # per-aspect normaliser = the root frequency (falls back to corpus size)
    root_freq: dict[str, int] = {}
    for aspect, root in GO_ROOTS.items():
        root_freq[aspect] = counts.get(root, n_sets) or n_sets

    ic: dict[str, float] = {}
    for t, c in counts.items():
        term_aspect = dag.aspect.get(t)
        denom = root_freq.get(term_aspect, n_sets) if term_aspect else n_sets
        p = c / denom
        ic[t] = float(-np.log(p)) if 0.0 < p < 1.0 else 0.0
    return ic


def _max_common_ancestor_ic(
    a_set: frozenset[str],
    b_set: frozenset[str],
    ic: Mapping[str, float],
) -> float:
    """IC of the most-informative common ancestor (the Resnik MICA score)."""
    common = a_set & b_set
    if not common:
        return 0.0
    return max((ic.get(t, 0.0) for t in common), default=0.0)


def resnik_pairwise(
    closures: Sequence[frozenset[str]],
    ic: Mapping[str, float],
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Resnik semantic similarity = IC(MICA) for each (i, j) index pair."""
    out = np.empty(len(pairs), dtype=np.float32)
    for n, (i, j) in enumerate(pairs):
        out[n] = _max_common_ancestor_ic(closures[i], closures[j], ic)
    return out


def lin_pairwise(
    closures: Sequence[frozenset[str]],
    ic: Mapping[str, float],
    pairs: Sequence[tuple[int, int]],
    best_ic: Sequence[float],
) -> np.ndarray:
    """Lin semantic similarity = 2*IC(MICA) / (IC(a)+IC(b)) for each index pair.

    ``best_ic[i]`` is the maximum IC over protein ``i``'s closure (its most specific
    annotation), the standard Lin denominator term.
    """
    out = np.empty(len(pairs), dtype=np.float32)
    for n, (i, j) in enumerate(pairs):
        mica = _max_common_ancestor_ic(closures[i], closures[j], ic)
        denom = best_ic[i] + best_ic[j]
        out[n] = (2.0 * mica / denom) if denom > 0.0 else 0.0
    return out
