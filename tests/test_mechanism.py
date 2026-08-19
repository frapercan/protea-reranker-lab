"""The mechanism space, and the streaming property that decides what can be deployed.

The plan is to refine on a bounded probe set and then run the winner over the whole
corpus without materialising intermediates. That makes streamability a property of
the science rather than of the engineering: a variant that wins the probe and
cannot be streamed is a result we cannot have, and finding that out afterwards is
the failure shape this project keeps meeting.

So the load-bearing tests here are that the accumulator does not grow with protein
length and that consuming residues in chunks gives the same answer as consuming
them at once. Everything else is the axis catalogue behaving as described.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.mechanism import (
    MechanismSpec,
    VariantGrid,
    enumerate_variants,
    expected_gated_support,
    is_streamable,
    probe_only,
    variant_name,
)
from protea_reranker_lab.mechanism_apply import (
    Accumulator,
    apply_mechanism,
    local_codes,
    peak_state_bytes,
)


def _spec(**kw) -> MechanismSpec:
    base = {"name": "", "dictionary": 64, "k_sequence": 8, **kw}
    spec = MechanismSpec(**base)
    return MechanismSpec(**{**spec.__dict__, "name": variant_name(spec)})


# ----------------------------------------------------------------- the streaming claim

def test_the_accumulator_does_not_grow_with_protein_length():
    """The whole corpus argument. If this fails the mechanism cannot be deployed."""
    spec = _spec(local_granularity="residue", k_local=8)

    assert peak_state_bytes(spec, 100) == peak_state_bytes(spec, 100_000)


def test_chunked_and_unchunked_agree():
    """Residues arrive in blocks in a corpus pass, so the reduction must be
    associative. A mechanism that depends on block boundaries would give a
    different answer per batch size and nothing would report it."""
    rng = np.random.default_rng(0)
    X = rng.random((400, 64)).astype(np.float32)
    spec = _spec(local_granularity="residue", k_local=8)

    whole = Accumulator(total=np.zeros(64), mode="mean")
    whole.absorb(local_codes(X, spec))

    in_pieces = Accumulator(total=np.zeros(64), mode="mean")
    for start in range(0, 400, 37):          # 37 divides nothing, on purpose
        in_pieces.absorb(local_codes(X[start:start + 37], spec))

    assert np.allclose(whole.emit(), in_pieces.emit())
    assert whole.count == in_pieces.count == 400
    assert np.array_equal(apply_mechanism(X, spec), apply_mechanism(X, spec))


def test_a_blocking_aggregator_is_refused_rather_than_approximated():
    """Attention cannot emit before it has seen every residue. Measured on the
    probe, yes; silently run at corpus scale, no."""
    spec = _spec(aggregator="attention")

    ok, reason = is_streamable(spec)

    assert ok is False
    assert "every residue" in reason


def test_applying_a_blocking_variant_raises():
    spec = _spec(aggregator="attention")

    with pytest.raises(ValueError, match="cannot be applied as a stream"):
        apply_mechanism(np.zeros((10, 64), dtype=np.float32), spec)


def test_the_reason_says_whether_the_limit_is_fundamental():
    """A reader must be able to tell 'impossible' from 'not built yet', because
    they cost differently."""
    _ok, reason = is_streamable(_spec(aggregator="attention"))

    assert "online-softmax" in reason


def test_a_gate_without_local_codes_is_refused():
    """There is nothing to gate, and with locals it needs two passes."""
    ok, reason = is_streamable(_spec(gate=True))

    assert ok is False
    assert "two passes" in reason


# ------------------------------------------------------------------ the axes behave

def test_every_variant_produces_a_k_sparse_code():
    rng = np.random.default_rng(1)
    X = rng.random((300, 64)).astype(np.float32)

    for gran, kl in [("none", None), ("cell", 8), ("residue", 8)]:
        for agg in ("mean", "sum", "max"):
            spec = _spec(local_granularity=gran, k_local=kl, aggregator=agg)
            assert int((apply_mechanism(X, spec) != 0).sum()) == 8


def test_frequency_discards_magnitude_and_magnitude_keeps_it():
    """The contrast that decides whether the historical negative was the binary
    code rather than the order."""
    X = np.array([[9.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    mag = local_codes(X, _spec(local_granularity="residue", k_local=1))
    freq = local_codes(X, _spec(local_granularity="residue", k_local=1,
                                weighting="frequency"))

    assert mag[0, 0] == 9.0
    assert freq[0, 0] == 1.0


def test_a_single_layer_has_no_partition_to_make():
    """Partitioning one layer is the shared case, so it must not double the grid."""
    got = enumerate_variants(VariantGrid(
        layer_sets=((-1,),), layer_modes=("shared", "partitioned"),
        aggregators=("mean",), weightings=("magnitude",), normalizations=("raw",),
        granularities=("none",), k_locals=(32,), gates=(False,)))

    assert len(got) == 1


def test_a_partitioned_layer_set_widens_the_dictionary():
    """Each layer gets its own block, so the collision arithmetic moves with it."""
    spec = _spec(layers=(0, 10, 19), layer_mode="partitioned", dictionary=2048)

    assert spec.effective_dictionary == 6144


def test_a_collapsing_gate_is_dropped_from_the_grid():
    """2048 by 128 gated is eight atoms, which measures an empty vector."""
    got = enumerate_variants(VariantGrid(
        granularities=("residue",), aggregators=("mean",), weightings=("magnitude",),
        normalizations=("raw",), layer_sets=((-1,),), layer_modes=("shared",),
        k_locals=(128,), dictionary=2048, k_sequence=128, gates=(True,)))

    assert got == []


def test_expected_gated_support_follows_the_effective_dictionary():
    shared = _spec(layers=(0, 10), layer_mode="shared", dictionary=1024,
                   k_sequence=128, gate=True)
    partitioned = _spec(layers=(0, 10), layer_mode="partitioned", dictionary=1024,
                        k_sequence=128, gate=True)

    assert expected_gated_support(shared) == pytest.approx(16.0)
    assert expected_gated_support(partitioned) == pytest.approx(8.0)


def test_the_name_reconstructs_the_variant():
    """A result must never need a lookup table to say what produced it."""
    spec = _spec(local_granularity="residue", k_local=32, weighting="frequency",
                 aggregator="max", normalize="zscore", layers=(0, 10),
                 layer_mode="partitioned")

    assert spec.name == "residue.kl32.freq.max.zscore.L0-10.part"


def test_the_grid_separates_what_can_be_deployed_from_what_cannot():
    variants = enumerate_variants()
    blocked = probe_only(variants)

    assert blocked
    assert all(not is_streamable(s)[0] for s in blocked)
    assert all(is_streamable(s)[0] for s in variants if s not in blocked)
