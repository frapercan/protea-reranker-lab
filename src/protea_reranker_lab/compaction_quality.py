"""Compaction-quality primitives: how well does a per-protein code preserve the
*full* per-residue signal, at a given storage budget, across all protein lengths?

The question is NOT downstream task accuracy (it saturates). Forward is cheap, so a
protein's per-residue token set is computed transiently; what we *cannot* afford is
storing the dense per-residue tensor (about 283 GB over the corpus). The compaction
of that per-residue signal is therefore the science, and a sparse code is one
candidate compaction among several dense controls.

Two reference ("gold") notions of full per-residue similarity that a compaction must
approximate:

* :func:`late_interaction_maxsim` -- the primary gold. Symmetric mean-of-max cosine
  over the two residue token sets (the ColBERT late-interaction kernel applied to
  protein residues).
* :func:`sinkhorn_ot_sim` -- a control gold. Entropic-regularised optimal transport
  (Sinkhorn) over the residue sets, a robustness check that the verdict is not an
  artefact of the max-sim kernel.

Candidate compactions (UNSUPERVISED in this pass, to isolate compaction from
learning), each exposing a fixed ``bytes_per_protein`` so a Pareto curve of
preservation-vs-bytes can be drawn:

* ``mean`` / ``max`` -- pool the residues to one ``d``-float dense vector (the
  classical dense controls); scored by cosine.
* ``pca_mean`` -- the mean vector projected to ``r`` dims by a PCA fit on the sample
  pool (a dense control at a reduced budget); scored by cosine.
* ``sdr_union`` -- per-residue k-WTA top-k, OR-bundled across residues into one binary
  Sparse Distributed Representation, then re-sparsified to the global top-k active
  dims (length-normalised by activation frequency); scored by Tanimoto. Stored as
  ``k`` uint16 indices.
* ``multivector`` -- keep ``M`` representative residue codes (k-means centroids over
  the residue cloud, each ``d`` floats) and score by late-interaction over the ``M``
  codes (a multi-vector dense control; the honest competitor to the single gold).

All functions are pure and torch-free at the boundary (numpy in / numpy out) except
the gold kernels, which take a torch device for the heavy pairwise reductions. Nothing
here reads post-cutoff data; the caller supplies the residue arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Length buckets (residues). Cutoffs co-designed with the operator; chosen so each
# bucket holds a comparable share of the annotated corpus and isolates the long /
# very-long regime where pooling is expected to lose the most signal.
BUCKET_EDGES = (318, 969, 1959)
BUCKET_NAMES = ("short", "medium", "long", "very_long")


def length_bucket(length: int) -> str:
    """Map a residue length to one of the four length buckets."""
    if length <= BUCKET_EDGES[0]:
        return BUCKET_NAMES[0]
    if length <= BUCKET_EDGES[1]:
        return BUCKET_NAMES[1]
    if length <= BUCKET_EDGES[2]:
        return BUCKET_NAMES[2]
    return BUCKET_NAMES[3]


# ---------------------------------------------------------------------------
# Gold similarities over residue token sets (the "full signal" references)
# ---------------------------------------------------------------------------
def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return (x / n).astype(np.float32)


def late_interaction_maxsim(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric mean-of-max cosine (ColBERT late interaction) over residue sets.

    ``p`` is ``(Lp, d)`` and ``q`` is ``(Lq, d)`` of residue vectors. Returns
    ``0.5 * (mean_i max_j cos(p_i, q_j) + mean_j max_i cos(p_i, q_j))``. This is the
    primary gold: the most faithful cheap notion of "how similar are these two
    proteins given every residue".
    """
    pu, qu = _l2norm_rows(p), _l2norm_rows(q)
    sim = pu @ qu.T  # (Lp, Lq) cosine matrix
    return float(0.5 * (sim.max(axis=1).mean() + sim.max(axis=0).mean()))


def sinkhorn_ot_sim(
    p: np.ndarray,
    q: np.ndarray,
    *,
    epsilon: float = 0.05,
    n_iter: int = 50,
) -> float:
    """Entropic-regularised optimal-transport similarity over residue sets (control).

    Cost is ``1 - cos`` between residue pairs; uniform marginals. Returns the negative
    transport cost (so higher = more similar), the standard OT analogue of the
    max-sim gold used to check the verdict is kernel-robust.
    """
    pu, qu = _l2norm_rows(p), _l2norm_rows(q)
    cost = 1.0 - (pu @ qu.T)  # (Lp, Lq) in [0, 2]
    lp, lq = cost.shape
    a = np.full(lp, 1.0 / lp, dtype=np.float32)
    b = np.full(lq, 1.0 / lq, dtype=np.float32)
    k = np.exp(-cost / epsilon).astype(np.float32) + 1e-30
    u = np.ones(lp, dtype=np.float32)
    v = np.ones(lq, dtype=np.float32)
    for _ in range(n_iter):
        u = a / (k @ v + 1e-30)
        v = b / (k.T @ u + 1e-30)
    plan = (u[:, None] * k) * v[None, :]
    transport_cost = float((plan * cost).sum())
    return -transport_cost


# ---------------------------------------------------------------------------
# Compactions: each returns a per-protein code; a sim fn scores a pair of codes
# ---------------------------------------------------------------------------
@dataclass
class Compaction:
    """A named compaction with its codes, a pairwise-sim fn, and its byte budget."""

    name: str
    budget_label: str
    bytes_per_protein: int
    sim: Callable[[int, int], float]


def mean_pool(residues: Sequence[np.ndarray]) -> np.ndarray:
    """Mean over residues -> ``(n, d)`` dense matrix."""
    return np.vstack([r.mean(axis=0) for r in residues]).astype(np.float32)


def max_pool(residues: Sequence[np.ndarray]) -> np.ndarray:
    """Coordinate-wise max over residues -> ``(n, d)`` dense matrix."""
    return np.vstack([r.max(axis=0) for r in residues]).astype(np.float32)


def pca_fit_transform(mean_mat: np.ndarray, r: int) -> np.ndarray:
    """PCA-project a ``(n, d)`` mean matrix to ``r`` components (fit on the pool).

    Transductive PCA on the sample pool (matches the lab's documented
    PCA-fit-on-FULL-pool convention). Returns ``(n, r)`` float32.
    """
    x = mean_mat.astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    # economy SVD; components are the right singular vectors
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:r]
    return (xc @ comps.T).astype(np.float32)


def cosine_pairwise_fn(codes: np.ndarray) -> Callable[[int, int], float]:
    """Return a pair-sim closure computing cosine on rows of a dense ``codes`` matrix."""
    unit = _l2norm_rows(codes)

    def _sim(i: int, j: int) -> float:
        return float(unit[i] @ unit[j])

    return _sim


def sdr_union_codes(
    residues: Sequence[np.ndarray],
    per_residue_k: int,
    final_k: int,
) -> tuple[np.ndarray, int]:
    """Per-residue k-WTA -> OR-bundle -> length-normalised global top-k re-sparsify.

    For each protein every residue contributes its ``per_residue_k`` top-magnitude
    dims; activations are accumulated across residues and divided by length (so a long
    protein does not trivially light up every dim), then the global top-``final_k``
    dims become the binary code. Returns ``(n, d)`` uint8 bitset and the byte cost
    (``final_k`` uint16 indices = ``2 * final_k`` bytes).
    """
    d = residues[0].shape[1]
    n = len(residues)
    out = np.zeros((n, d), dtype=np.uint8)
    for i, m in enumerate(residues):
        kk = min(per_residue_k, d - 1)
        idx = np.argpartition(-np.abs(m), kk, axis=1)[:, :kk]  # top-k dims per residue
        acc = np.zeros(d, dtype=np.float32)
        np.add.at(acc, idx.reshape(-1), 1.0)
        acc /= max(1, m.shape[0])  # length-normalise the activation frequency
        fk = min(final_k, d)
        top = np.argpartition(-acc, fk - 1)[:fk]
        out[i, top] = 1
    return out, 2 * final_k


def tanimoto_pairwise_fn(sdr: np.ndarray) -> Callable[[int, int], float]:
    """Return a pair-sim closure computing Tanimoto overlap on a uint8 bitset."""
    sdr_f = sdr.astype(np.float32)
    pop = sdr_f.sum(axis=1)

    def _sim(i: int, j: int) -> float:
        inter = float(sdr_f[i] @ sdr_f[j])
        denom = pop[i] + pop[j] - inter
        return inter / denom if denom > 0 else 0.0

    return _sim


def multivector_codes(
    residues: Sequence[np.ndarray],
    m: int,
    *,
    seed: int = 0,
) -> tuple[list[np.ndarray], int]:
    """Reduce each protein to ``M`` representative residue codes via k-means.

    Runs a light Lloyd k-means (few iters) over each protein's residue cloud and keeps
    the ``M`` centroids (l2-normalised). Proteins shorter than ``M`` keep all their
    residues. Returns the list of ``(<=M, d)`` code sets and the per-protein byte cost
    (``M * d`` float16 = ``2 * M * d`` bytes).
    """
    d = residues[0].shape[1]
    rng = np.random.default_rng(seed)
    codes: list[np.ndarray] = []
    for r in residues:
        ru = _l2norm_rows(r)
        if ru.shape[0] <= m:
            codes.append(ru.astype(np.float16).astype(np.float32))
            continue
        # k-means++ light init: random distinct rows
        cen = ru[rng.choice(ru.shape[0], size=m, replace=False)].copy()
        for _ in range(8):
            sims = ru @ cen.T  # cosine (rows are unit)
            assign = sims.argmax(axis=1)
            for c in range(m):
                members = ru[assign == c]
                if members.shape[0]:
                    cen[c] = members.mean(axis=0)
            cen = _l2norm_rows(cen)
        codes.append(cen.astype(np.float16).astype(np.float32))
    return codes, 2 * m * d


def multivector_pairwise_fn(
    codes: Sequence[np.ndarray],
) -> Callable[[int, int], float]:
    """Late-interaction (mean-of-max cosine) over two ``M``-code sets."""

    def _sim(i: int, j: int) -> float:
        return late_interaction_maxsim(codes[i], codes[j])

    return _sim


# ---------------------------------------------------------------------------
# Selection-without-matching: collapse the SAME M representatives to a single
# vector. These decompose the multi-vector edge into (a) selecting M denoised
# modes and (b) matching them via late interaction. Each single-vector code
# below keeps the SELECTION and DROPS the MATCHING, so comparing them against
# ``mean`` (mean of ALL residues) and ``multivector`` (full late interaction)
# isolates which of the two carries the multi-vector advantage.
# ---------------------------------------------------------------------------
def representative_residues(
    residues: Sequence[np.ndarray],
    m: int,
    *,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return the SAME ``M`` representative residue centroids the multi-vector arm keeps.

    Identical light Lloyd k-means selection as :func:`multivector_codes`, but the
    centroids are returned UN-normalised (raw cluster means in the residue space) and
    in float32, so a downstream single-vector collapse can keep magnitude information
    (e.g. norm-weighting). Proteins with at most ``M`` residues keep all their residues.
    The returned cloud is the selection that the multi-vector arm matches over.
    """
    rng = np.random.default_rng(seed)
    reps: list[np.ndarray] = []
    for r in residues:
        x = r.astype(np.float32)
        if x.shape[0] <= m:
            reps.append(x.copy())
            continue
        ru = _l2norm_rows(x)  # assign on cosine, same as the multi-vector k-means
        cen_unit = ru[rng.choice(ru.shape[0], size=m, replace=False)].copy()
        assign = np.zeros(ru.shape[0], dtype=np.int64)
        for _ in range(8):
            assign = (ru @ cen_unit.T).argmax(axis=1)
            for c in range(m):
                members = ru[assign == c]
                if members.shape[0]:
                    cen_unit[c] = members.mean(axis=0)
            cen_unit = _l2norm_rows(cen_unit)
        # raw (un-normalised) centroids = mean of the ORIGINAL residues per cluster,
        # so cluster magnitude (a denoised "salience") survives for norm-weighting.
        cen_raw = np.zeros((m, x.shape[1]), dtype=np.float32)
        for c in range(m):
            members = x[assign == c]
            cen_raw[c] = members.mean(axis=0) if members.shape[0] else cen_unit[c]
        reps.append(cen_raw)
    return reps


def mean_of_reps(reps: Sequence[np.ndarray]) -> np.ndarray:
    """Mean over each protein's ``M`` representative centroids -> ``(n, d)`` dense.

    Selection (the M denoised modes) plus a plain collapse, with NO late-interaction
    matching: the cheap single-vector code that keeps the multi-vector SELECTION.
    """
    return np.vstack([rep.mean(axis=0) for rep in reps]).astype(np.float32)


def normweighted_mean_of_reps(reps: Sequence[np.ndarray]) -> np.ndarray:
    """Norm-weighted mean over the ``M`` representatives -> ``(n, d)`` dense.

    Weights each representative by its L2 norm (a cheap unsupervised "salience" proxy
    for the residue modes; NOT learned attention). Still a single d-vector with no
    matching, so it isolates whether weighting the selection helps over a flat mean.
    """
    out = []
    for rep in reps:
        w = np.linalg.norm(rep, axis=1)
        s = w.sum()
        if s <= 0:
            out.append(rep.mean(axis=0))
        else:
            out.append((rep * w[:, None]).sum(axis=0) / s)
    return np.vstack(out).astype(np.float32)


def sdr_union_of_reps(
    reps: Sequence[np.ndarray],
    per_residue_k: int,
    final_k: int,
) -> tuple[np.ndarray, int]:
    """SDR-union over the ``M`` representatives instead of all ``L`` residues.

    Same k-WTA OR-bundle + length-normalised global top-``final_k`` re-sparsify as
    :func:`sdr_union_codes`, but the bundle is built from the M representative
    centroids only (the selection), so it tests whether a sparse code over the
    denoised modes beats one over the raw residue cloud. Returns the ``(n, d)`` uint8
    bitset and the byte cost (``final_k`` uint16 indices = ``2 * final_k`` bytes).
    """
    return sdr_union_codes(reps, per_residue_k, final_k)


__all__ = [
    "BUCKET_EDGES",
    "BUCKET_NAMES",
    "Compaction",
    "cosine_pairwise_fn",
    "late_interaction_maxsim",
    "length_bucket",
    "max_pool",
    "mean_of_reps",
    "mean_pool",
    "multivector_codes",
    "multivector_pairwise_fn",
    "normweighted_mean_of_reps",
    "pca_fit_transform",
    "representative_residues",
    "sdr_union_codes",
    "sdr_union_of_reps",
    "sinkhorn_ot_sim",
    "tanimoto_pairwise_fn",
]
