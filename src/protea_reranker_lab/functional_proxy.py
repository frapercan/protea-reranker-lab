"""The cheap screen that predicted the real task once, rebuilt, with its own confound closed.

The layer ablation screened representations on a functional-gold proxy before
spending anything on real kNN transfer, and the proxy PREDICTED the real winner:
it ranked an early-mid layer with standardisation and k-WTA above the last-layer
dense mean-pool, and the board-faithful confirm then reproduced that ordering. So
the proxy is not a convenience, it is a screen with one recorded success.

Two measures, as before:

* the global rank agreement between code cosine and propagated-GO similarity over
  sampled pairs, which says whether the geometry means anything at all
* neighbour purity, the mean functional similarity to the nearest neighbours,
  which is closer to what kNN transfer actually consumes

THE CONFOUND THIS ADDS, WHICH THE ORIGINAL DID NOT HAVE TO FACE

That screen compared dense and lightly sparsified representations. This one is
sweeping sparsity itself, and cosine between sparse codes is degenerate in a way
that scales with sparsity: two independent k-sparse codes over a D-atom dictionary
share about k squared over D atoms, so at k=128 and D=2048 they overlap in eight,
and at k=16 they overlap in one eighth of a pair and MOST PAIRS HAVE COSINE
EXACTLY ZERO.

A rank correlation over a variable that is mostly ties is not comparable to one
over a continuous variable, so a sparser arm can score differently for a reason
that is arithmetic rather than functional. ``zero_pair_fraction`` is reported
beside every score and ``refuse_a_degenerate_screen`` stops a comparison whose
ties dominate, because a screen that silently rewards or punishes sparsity would
corrupt the one axis the whole study is about.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: Above this share of exactly-zero cosines the rank statistic is describing the
#: tie structure rather than the geometry.
MAX_TIED_PAIR_FRACTION = 0.5


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def pair_cosines(codes: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine for each (i, j) row of ``pairs``, zero when either code is empty."""
    left, right = codes[pairs[:, 0]], codes[pairs[:, 1]]
    num = (left * right).sum(axis=1)
    den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def zero_pair_fraction(cosines: np.ndarray) -> float:
    """Share of pairs whose codes share no atom at all.

    The quantity that makes a rank statistic incomparable across sparsity levels,
    reported so a reader can see whether a difference is geometric or arithmetic.
    """
    return float(np.mean(cosines == 0.0)) if cosines.size else 0.0


def refuse_a_degenerate_screen(cosines: np.ndarray,
                               limit: float = MAX_TIED_PAIR_FRACTION) -> None:
    """Stop a screen whose pairs are mostly tied at zero.

    Refused rather than warned because the number would still be produced, would
    still look like a score, and would be compared against arms whose ties are
    fewer. The fix is more pairs, a wider k, or a narrower dictionary, and all
    three are decisions rather than accidents.
    """
    share = zero_pair_fraction(cosines)
    if share <= limit:
        return
    raise ValueError(
        f"{share:.1%} of sampled pairs share no atom, above the {limit:.0%} at "
        "which a rank correlation describes the tie structure rather than the "
        "geometry. Raise k, shrink the dictionary, or sample pairs among "
        "functional neighbours rather than uniformly"
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, ties averaged, without pulling in scipy for one number."""
    if x.size < 2:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # average the ranks of tied values, which is what makes this Spearman rather
    # than a correlation of arbitrary orderings
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def refuse_a_constant_gold(gold: np.ndarray, min_distinct: int = 3) -> None:
    """Stop a screen whose functional similarities barely vary.

    The mirror of the tie problem, and it is created by the fix for it. Sampling
    only pairs above a similarity threshold removes the low end, and where the
    similarity distribution is clustered, as it is when a pool has clean families,
    every surviving pair can carry nearly the same value. A rank correlation
    against a constant is zero however good the geometry is, so the screen would
    report a null for a sampling reason.

    Both guards together say the screen needs pairs that vary on BOTH sides:
    enough shared atoms for cosine to be informative, and enough spread in
    function for there to be an ordering to recover.
    """
    distinct = int(np.unique(np.round(gold, 6)).size)
    if distinct >= min_distinct:
        return
    raise ValueError(
        f"the sampled pairs carry only {distinct} distinct functional "
        "similarities, so there is no ordering for a rank correlation to recover. "
        "Lower min_similarity, or sample across a graded neighbourhood rather than "
        "a thresholded one"
    )


def rank_agreement(codes: np.ndarray, closures: list[frozenset[str]],
                   pairs: np.ndarray, *, strict: bool = True) -> dict:
    """Spearman between code cosine and functional similarity, with both guards.

    Reports the tie share and the number of distinct gold values, because a null
    from either degeneracy looks exactly like a null from the representation, and
    those are the two ways this screen can lie.
    """
    cos = pair_cosines(codes, pairs)
    gold = np.array([jaccard(closures[i], closures[j]) for i, j in pairs])
    if strict:
        refuse_a_degenerate_screen(cos)
        refuse_a_constant_gold(gold)
    return {
        "spearman": spearman(cos, gold),
        "zero_pair_fraction": zero_pair_fraction(cos),
        "distinct_gold_values": int(np.unique(np.round(gold, 6)).size),
        "pairs": int(pairs.shape[0]),
    }


def neighbour_purity(codes: np.ndarray, closures: list[frozenset[str]], k: int) -> dict:
    """Mean functional similarity to the k nearest neighbours, excluding self.

    Closer to what kNN transfer consumes than a global rank statistic, and it is
    the measure on which the recorded screen and the real task agreed.
    """
    norms = np.linalg.norm(codes, axis=1, keepdims=True)
    unit = codes / np.where(norms > 0, norms, 1.0)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    scores = [
        float(np.mean([jaccard(closures[i], closures[j]) for j in top[i]]))
        for i in range(codes.shape[0])
    ]
    return {"purity": float(np.mean(scores)), "k": k, "n": len(scores)}


def sample_pairs(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform unordered pairs without self-pairs, as an (count, 2) index array."""
    i = rng.integers(0, n, size=count)
    j = rng.integers(0, n, size=count)
    keep = i != j
    return np.stack([i[keep], j[keep]], axis=1)


def sample_neighbour_pairs(closures: list[frozenset[str]], count: int,
                           rng: np.random.Generator, *,
                           min_similarity: float = 0.05,
                           attempts_per_pair: int = 20) -> np.ndarray:
    """Pairs that share some function, which is the regime kNN transfer lives in.

    Uniform pairs are mostly functionally unrelated proteins, and at high sparsity
    those are exactly the pairs whose codes share no atom, so a uniform screen
    measures the tie structure instead of the geometry. Biasing toward pairs with
    a functional relationship keeps the screen usable where the interesting
    sparsity is, and is also closer to the question: a retrieval system is judged
    on how it orders plausible neighbours, not on how confidently it separates two
    unrelated proteins.

    This changes what the number means, so it is a separate function rather than a
    flag. A Spearman over neighbour-biased pairs and one over uniform pairs are not
    comparable, and giving them one name would let them be averaged.
    """
    n = len(closures)
    out: list[tuple[int, int]] = []
    for _ in range(count * attempts_per_pair):
        if len(out) >= count:
            break
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        if jaccard(closures[i], closures[j]) >= min_similarity:
            out.append((i, j))
    if not out:
        raise ValueError(
            f"no pair reached similarity {min_similarity} in "
            f"{count * attempts_per_pair} attempts. Either the closures are empty "
            "or the pool has no functional structure to screen on"
        )
    if len(out) < count:
        log.warning("neighbour sampling produced %d of %d requested pairs",
                    len(out), count)
    return np.array(out, dtype=np.int64)

