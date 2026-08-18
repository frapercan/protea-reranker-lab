"""Making the dictionary usable as a retrieval index, and measuring whether it is.

The served encoder is trained so the cosine between two codes matches functional
similarity. Nothing in that objective asks its atoms to be DISCRIMINATIVE, and the
consequence is measurable on the shipped bank: one atom of 2048 sits in half of
575,503 rows, 385 atoms sit in more than a tenth, and the ten longest posting lists
carry 0.69 to 1.11 nats against a median atom's 3.18.

That is not a defect in the training, it is a stage nobody asked anything of. A
retrieval index built on such a dictionary spends its work on atoms that separate
nothing, which is why pruning the query to its informative atoms cuts work by 14x
while exact top-10 collapses: the informative atoms were never made to carry the
neighbourhood on their own.

This module holds the term that asks for it, and the measurements that say whether
the ask worked. The measurements matter as much as the term: a usage penalty can
flatten atom frequency while destroying the geometry, and the only way to tell is
to keep both numbers side by side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


def load_balance_penalty(activations):
    """Penalty that is 1.0 when every atom carries equal mass and D when one carries all.

    ``D * sum(p^2)`` over the normalised mean absolute activation. Differentiable,
    cheap, and it acts on the quantity that top-k selection reads: an atom with no
    magnitude anywhere is never selected, and an atom with magnitude everywhere is
    always selected and therefore separates nothing.

    It is deliberately NOT a penalty on the post-top-k document frequency, which is
    what we actually care about and is not differentiable. That gap is the reason
    the effect has to be measured after training rather than assumed from the loss.
    """
    mass = activations.abs().mean(dim=0)
    p = mass / (mass.sum() + 1e-9)
    return p.numel() * (p * p).sum()


def effective_width(codes_idx: np.ndarray, dictionary: int) -> dict:
    """How many atoms the code really uses, as the exponential of usage entropy.

    A dictionary of 2048 whose usage entropy gives exp(H) = 548 is a 548-atom
    dictionary wearing a larger label, and collision arithmetic computed on the
    nominal width understates overlap by the ratio between them.
    """
    counts = np.bincount(codes_idx.ravel(), minlength=dictionary).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("no atoms are used at all; the code is empty")
    p = counts / total
    nz = p > 0
    entropy = float(-(p[nz] * np.log(p[nz])).sum())
    return {
        "entropy_nats": entropy,
        "effective_width": float(np.exp(entropy)),
        "nominal_width": dictionary,
        "used_fraction": float(np.exp(entropy) / dictionary),
        "dead_atoms": int((counts == 0).sum()),
    }


def posting_lengths(codes_idx: np.ndarray, dictionary: int) -> dict:
    """The shape of the index this dictionary would build.

    The head is what decides whether retrieval can prune: an atom present in half
    the bank puts half the bank in every candidate set that touches it, whatever
    the rest of the code does.
    """
    n = codes_idx.shape[0]
    # DISTINCT rows per atom, not total occurrences. A posting list holds each row
    # once however many times the atom appears in it, and counting occurrences let
    # a share exceed 1.0 on a code with a repeated atom, which is nonsense a reader
    # would have to notice rather than be told.
    flat_rows = np.repeat(np.arange(n, dtype=np.int64), codes_idx.shape[1])
    pairs = np.unique(np.stack([codes_idx.ravel().astype(np.int64), flat_rows]), axis=1)
    df = np.bincount(pairs[0], minlength=dictionary).astype(np.int64)
    share = df / n
    p = np.where(df > 0, share, 1.0)
    idf = -np.log(p)
    order = np.argsort(-df)
    return {
        "longest_share": float(share[order[0]]),
        "top10_shares": [float(share[a]) for a in order[:10]],
        "top10_idf": [float(idf[a]) for a in order[:10]],
        "median_idf": float(np.median(idf[df > 0])),
        "atoms_over_half": int((share > 0.5).sum()),
        "atoms_over_tenth": int((share > 0.1).sum()),
        "dead_atoms": int((df == 0).sum()),
    }


@dataclass(frozen=True)
class RetrievalProbe:
    """What a two-stage measurement is run against. Named so a result carries it."""

    queries: np.ndarray
    keep: int
    shortlist: int = 1000
    truth_k: int = 10


def two_stage_recall(codes_idx: np.ndarray, codes_val: np.ndarray, dictionary: int,
                     probe: RetrievalProbe) -> dict:
    """Work and recall for a cheap sparse pass followed by an exact rescore.

    Query atoms are kept in decreasing information, since the cost of an atom is
    its posting list and its value is its rarity. Work is counted in postings
    touched, which is what the scan actually pays; the candidate fraction saturates
    as soon as one atom is shared and cannot see a reduction at all, which is how a
    previous reading concluded the index prunes nothing.

    Recall is of the TRUE top-``truth_k`` inside the shortlist, because a second
    exact pass over the shortlist recovers the order. Exact top-k agreement under a
    pruned query measures a system nobody would build.
    """
    n, k = codes_idx.shape
    df = np.bincount(codes_idx.ravel(), minlength=dictionary).astype(np.int64)
    idf = -np.log(np.where(df > 0, df / n, 1.0))

    order = np.argsort(codes_idx.ravel(), kind="stable")
    rows = (order // k).astype(np.int32)
    vals = codes_val.ravel()[order]
    starts = np.searchsorted(codes_idx.ravel()[order], np.arange(dictionary + 1))

    def accumulate(atoms):
        acc = np.zeros(n, dtype=np.float32)
        work = 0
        for atom, value in atoms:
            s, e = starts[atom], starts[atom + 1]
            np.add.at(acc, rows[s:e], vals[s:e] * value)
            work += e - s
        return acc, work

    recalls, works, full_work = [], [], 0
    for q in probe.queries:
        atoms = list(zip(codes_idx[q], codes_val[q]))
        exact, w_full = accumulate(atoms)
        exact[q] = -np.inf
        top = np.argpartition(-exact, probe.truth_k)[:probe.truth_k]
        truth = set(top.tolist())
        full_work += w_full

        pruned = sorted(atoms, key=lambda t: -idf[t[0]])[:probe.keep]
        acc, w = accumulate(pruned)
        acc[q] = -np.inf
        # Clamped: a shortlist wider than the bank is a request for everything,
        # and argpartition raises rather than saying so.
        width = min(probe.shortlist, n - 1)
        short = np.argpartition(-acc, width)[:width]
        recalls.append(len(truth & set(short.tolist())) / probe.truth_k)
        works.append(w)

    mean_work = float(np.mean(works))
    return {
        "keep": probe.keep,
        "shortlist": probe.shortlist,
        "postings": mean_work,
        "speedup": float(full_work / len(probe.queries) / mean_work) if mean_work else 0.0,
        "recall": float(np.mean(recalls)),
        "queries": len(probe.queries),
    }
