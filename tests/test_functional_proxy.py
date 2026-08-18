"""The cheap screen, and the confound that would have corrupted the sparsity axis.

This screen has one recorded success: it ranked an early-mid layer with
standardisation and k-WTA above the last-layer dense mean-pool, and the
board-faithful kNN confirm then reproduced that ordering. So it earns its place as
a screen rather than a convenience.

But it was used on dense and lightly sparsified representations, and this study
sweeps sparsity itself. Cosine between sparse codes is degenerate in a way that
scales with sparsity: two independent k-sparse codes over D atoms share about
k squared over D, so most pairs eventually share nothing and the rank statistic
starts describing the tie structure instead of the geometry. A screen that
silently rewards or punishes sparsity would corrupt the one axis the whole study
is about, so the tie share is measured, a degenerate screen is refused, and a
sampling that survives the sparse regime is provided rather than only described.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.functional_proxy import (
    MAX_TIED_PAIR_FRACTION,
    jaccard,
    neighbour_purity,
    pair_cosines,
    rank_agreement,
    refuse_a_constant_gold,
    refuse_a_degenerate_screen,
    sample_neighbour_pairs,
    sample_pairs,
    spearman,
    zero_pair_fraction,
)
from protea_reranker_lab.sparse_aggregation import topk_real


def _structured_pool(n=200, d=512, families=8, seed=0):
    """A pool with GRADED functional structure, so a screen has an ordering to find.

    Clean disjoint families are the wrong fixture: every within-family pair carries
    the same similarity, so a threshold-sampled screen sees a constant gold and a
    rank correlation is zero however good the geometry is. Each protein here takes
    a variable number of its family's terms, which spreads the similarities.
    """
    rng = np.random.default_rng(seed)
    fams = rng.integers(0, families, size=n)
    shared = rng.integers(2, 9, size=n)
    closures = [
        frozenset({f"GO:{fams[i]}-{t}" for t in range(int(shared[i]))})
        | {f"GO:own-{i}"}
        for i in range(n)
    ]
    base = rng.random((families, d)).astype(np.float32)
    codes = base[fams] + 0.35 * rng.random((n, d)).astype(np.float32)
    return codes, closures, rng


# ------------------------------------------------------------------------ basics

def test_jaccard_is_zero_for_disjoint_and_one_for_identical():
    assert jaccard(frozenset("ab"), frozenset("cd")) == 0.0
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0


def test_jaccard_of_two_empty_sets_is_zero_rather_than_undefined():
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_spearman_is_one_on_a_monotone_relation():
    x = np.array([1.0, 2.0, 3.0, 4.0])

    assert spearman(x, 2 * x + 1) == pytest.approx(1.0)


def test_spearman_averages_tied_ranks():
    """Without tie averaging this is a correlation of arbitrary orderings."""
    x = np.array([1.0, 1.0, 2.0, 2.0])
    y = np.array([5.0, 5.0, 9.0, 9.0])

    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_of_a_constant_is_zero_rather_than_nan():
    assert spearman(np.ones(10), np.arange(10.0)) == 0.0


# --------------------------------------------------------------------- the confound

def test_the_tie_share_grows_as_the_code_gets_sparser():
    """The confound, demonstrated. Not a worry, a measured property of the metric."""
    codes, _closures, rng = _structured_pool(d=2048)
    pairs = sample_pairs(codes.shape[0], 8000, rng)

    shares = [zero_pair_fraction(pair_cosines(topk_real(codes, k), pairs))
              for k in (256, 64, 16)]

    assert shares[0] <= shares[1] <= shares[2]
    assert shares[2] > shares[0]


def test_a_degenerate_screen_is_refused():
    cosines = np.zeros(100)

    with pytest.raises(ValueError, match="share no atom"):
        refuse_a_degenerate_screen(cosines)


def test_the_refusal_offers_the_three_ways_out():
    """A guard that only refuses becomes a guard someone disables."""
    with pytest.raises(ValueError) as excinfo:
        refuse_a_degenerate_screen(np.zeros(10))

    message = str(excinfo.value)
    assert "Raise k" in message and "functional neighbours" in message


def test_a_healthy_screen_passes():
    refuse_a_degenerate_screen(np.linspace(0.1, 1.0, 100))


def test_the_limit_is_where_ties_take_over():
    assert 0.0 < MAX_TIED_PAIR_FRACTION < 1.0


# ------------------------------------------------------- the sampling that survives

def test_neighbour_biased_pairs_keep_the_screen_usable_where_uniform_cannot():
    """The regression this exists for: at k=32 the uniform screen is refused and
    the neighbour-biased one is not, on the same codes and the same pool."""
    codes, closures, rng = _structured_pool(d=2048)
    sparse = topk_real(codes, 32)

    uniform = pair_cosines(sparse, sample_pairs(codes.shape[0], 8000, rng))
    biased = pair_cosines(sparse, sample_neighbour_pairs(closures, 4000, rng))

    assert zero_pair_fraction(uniform) > MAX_TIED_PAIR_FRACTION
    assert zero_pair_fraction(biased) < MAX_TIED_PAIR_FRACTION


def test_neighbour_pairs_all_share_some_function():
    _codes, closures, rng = _structured_pool()

    pairs = sample_neighbour_pairs(closures, 200, rng, min_similarity=0.1)

    assert all(jaccard(closures[i], closures[j]) >= 0.1 for i, j in pairs)


def test_a_pool_with_no_structure_stops_rather_than_returning_nothing():
    """An empty pair array would sail on and produce a score over no data."""
    closures = [frozenset({f"GO:only-{i}"}) for i in range(50)]
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="no functional structure"):
        sample_neighbour_pairs(closures, 100, rng, min_similarity=0.5)


def test_the_two_samplings_are_separate_functions_on_purpose():
    """Their numbers are not comparable, and one name would let them be averaged."""
    assert sample_pairs is not sample_neighbour_pairs


# ------------------------------------------------------------------- the measures

def test_rank_agreement_finds_structure_when_it_is_there():
    codes, closures, rng = _structured_pool()
    pairs = sample_neighbour_pairs(closures, 3000, rng)

    got = rank_agreement(codes, closures, pairs)

    assert got["spearman"] > 0.0
    assert got["pairs"] == pairs.shape[0]


def test_rank_agreement_reports_the_tie_share_beside_the_score():
    """So a reader can tell a geometric difference from an arithmetic one."""
    codes, closures, rng = _structured_pool()
    pairs = sample_neighbour_pairs(closures, 2000, rng)

    assert "zero_pair_fraction" in rank_agreement(codes, closures, pairs)


def test_purity_is_higher_on_structured_codes_than_on_noise():
    codes, closures, rng = _structured_pool()
    noise = rng.random(codes.shape).astype(np.float32)

    assert (neighbour_purity(codes, closures, 5)["purity"]
            > neighbour_purity(noise, closures, 5)["purity"])


def test_purity_excludes_the_protein_itself():
    """Self is at cosine 1 and would otherwise dominate every neighbourhood, which
    is the same mistake as measuring best-donor identity against the query."""
    codes, closures, _rng = _structured_pool(n=20, d=64)

    got = neighbour_purity(codes, closures, 1)

    assert got["purity"] < 1.0


def test_a_constant_gold_is_refused_too():
    """The mirror of the tie problem, created by the fix for it: threshold sampling
    on a clustered similarity distribution leaves every pair carrying one value."""
    with pytest.raises(ValueError, match="no ordering"):
        refuse_a_constant_gold(np.full(100, 0.75))


def test_the_constant_gold_refusal_names_the_sampling_as_the_cause():
    with pytest.raises(ValueError) as excinfo:
        refuse_a_constant_gold(np.full(50, 0.5))

    assert "min_similarity" in str(excinfo.value)


def test_a_graded_gold_passes():
    refuse_a_constant_gold(np.linspace(0.1, 0.9, 40))


def test_rank_agreement_reports_the_gold_spread_as_well():
    """Both degeneracies produce a null that looks like a representation null, so
    both counts travel with the score."""
    codes, closures, rng = _structured_pool()
    pairs = sample_neighbour_pairs(closures, 2000, rng)

    got = rank_agreement(codes, closures, pairs)

    assert got["distinct_gold_values"] >= 3

