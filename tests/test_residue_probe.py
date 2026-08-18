"""Choosing and sizing the bounded probe, before any forward pass runs.

The residue tensor only ever has to exist for a probe, and the probe is where the
mechanism is refined. Two things decide whether that works: the sample has power
in the bands where the mechanism is expected to act, and the request is known to
fit before an hour of forward passes discovers that it does not.

Float16 is not an option and that is measured, not stylistic: layer 38 of
ankh-base reached a maximum absolute value of 440,611 against float16's 65,504,
and storing it there once produced a purity of 0.076, which is what random noise
gives.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.residue_probe import (
    LENGTH_BANDS,
    ProbeSpec,
    band_of,
    estimate_bytes,
    plan_probe,
    refuse_an_oversized_probe,
    select_stratified_probe,
)


def _pool(counts=(40, 30, 20, 10), seed=0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    lengths: dict[str, int] = {}
    for (name, low, high), n in zip(LENGTH_BANDS, counts, strict=True):
        top = min(high, 6000)
        for i in range(n):
            lengths[f"{name}-{i:04d}"] = int(rng.integers(low, top + 1))
    return lengths


# --------------------------------------------------------------------------- bands

def test_every_length_lands_in_exactly_one_band():
    assert band_of(1) == "<=512"
    assert band_of(512) == "<=512"
    assert band_of(513) == "512-1024"
    assert band_of(2048) == "1024-2048"
    assert band_of(2049) == ">2048"


def test_an_absurd_length_still_lands_somewhere():
    """A protein longer than any band's ceiling must not fall out of the sample
    silently, which is how a stratum quietly loses its tail."""
    assert band_of(10 ** 8) == ">2048"


# ------------------------------------------------------------------------ selection

def test_the_sample_is_equal_per_band_rather_than_proportional():
    """Power where the mechanism is expected to act, not where the corpus is
    thickest. The measured gain concentrated on long proteins."""
    chosen = select_stratified_probe(_pool(), ProbeSpec(per_band=10))

    assert {k: len(v) for k, v in chosen.items()} == {n: 10 for n, _, _ in LENGTH_BANDS}


def test_a_thin_band_yields_what_it_has_rather_than_failing():
    chosen = select_stratified_probe(_pool(counts=(40, 30, 20, 3)), ProbeSpec(per_band=10))

    assert len(chosen[">2048"]) == 3


def test_the_draw_is_reproducible():
    a = select_stratified_probe(_pool(), ProbeSpec(per_band=10, seed=7))
    b = select_stratified_probe(_pool(), ProbeSpec(per_band=10, seed=7))

    assert a == b


def test_a_different_seed_draws_a_different_probe():
    a = select_stratified_probe(_pool(), ProbeSpec(per_band=10, seed=7))
    b = select_stratified_probe(_pool(), ProbeSpec(per_band=10, seed=8))

    assert a != b


def test_the_pool_is_ordered_before_sampling():
    """Sampling an unordered collection with a fixed seed selects a different set
    every run and nothing downstream can tell. Same defect as the reference draw."""
    pool = _pool()
    shuffled = dict(reversed(list(pool.items())))

    assert (select_stratified_probe(pool, ProbeSpec(per_band=10))
            == select_stratified_probe(shuffled, ProbeSpec(per_band=10)))


# -------------------------------------------------------------------------- sizing

def test_the_size_is_summed_over_proteins_not_derived_from_a_mean():
    """Pushing a summary through a nonlinear function is how a storage figure came
    out wrong twice already."""
    lengths = {"a": 100, "b": 300}
    chosen = {"<=512": ["a", "b"]}
    spec = ProbeSpec(layers=(0, 1), width=10)

    assert estimate_bytes(lengths, chosen, spec) == (100 + 300) * 2 * 10 * 4


def test_truncation_is_counted_in_the_size():
    lengths = {"a": 5000}
    chosen = {">2048": ["a"]}
    spec = ProbeSpec(layers=(0,), width=10, max_length=2048)

    assert estimate_bytes(lengths, chosen, spec) == 2048 * 10 * 4


def test_an_oversized_probe_is_refused_before_the_forward_passes():
    with pytest.raises(ValueError, match="over the"):
        refuse_an_oversized_probe(9 * 1024 ** 3, ProbeSpec(max_bytes=8 * 1024 ** 3))


def test_the_refusal_says_float16_is_not_a_way_out():
    """It is the obvious saving and it silently corrupts middle layers."""
    with pytest.raises(ValueError) as excinfo:
        refuse_an_oversized_probe(9 * 1024 ** 3, ProbeSpec(max_bytes=1))

    assert "float16" in str(excinfo.value)


def test_a_probe_that_fits_passes():
    refuse_an_oversized_probe(1024, ProbeSpec(max_bytes=8 * 1024 ** 3))


# ---------------------------------------------------------------------- the plan

def test_the_plan_reports_which_proteins_will_be_truncated():
    """The long band is where the mechanism is expected to act most and is exactly
    the band that gets cut, so the count must be on the plan rather than inferred."""
    lengths = _pool()
    plan = plan_probe(lengths, ProbeSpec(per_band=5, max_length=2048))

    assert set(plan["truncated"]) == {
        a for members in plan["accessions"].values() for a in members
        if lengths[a] > 2048
    }


def test_the_plan_costs_nothing_when_it_is_rejected():
    """Selection and sizing happen before any model is touched, so a refused plan
    is free rather than an hour wasted."""
    with pytest.raises(ValueError):
        plan_probe(_pool(), ProbeSpec(per_band=40, layers=tuple(range(49)),
                                      max_bytes=1024))


def test_the_plan_carries_the_layers_it_was_sized_for():
    plan = plan_probe(_pool(), ProbeSpec(per_band=5, layers=(0, 10)))

    assert plan["layers"] == [0, 10]
