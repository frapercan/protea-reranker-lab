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
    chunk_spans,
    computed_residues,
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


def test_a_long_protein_is_sized_at_its_full_length():
    """Chunked, not truncated: every residue is stored, so the size is the whole
    protein rather than a window of it."""
    lengths = {"a": 5000}
    chosen = {">2048": ["a"]}
    spec = ProbeSpec(layers=(0,), width=10)

    assert estimate_bytes(lengths, chosen, spec) == 5000 * 10 * 4


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

def test_nothing_is_truncated_and_the_chunked_ones_are_named():
    """The long band is where the mechanism is expected to act most, so cutting it
    would measure the mechanism on a prefix precisely where it matters."""
    lengths = _pool()
    plan = plan_probe(lengths, ProbeSpec(per_band=5, chunk_size=1024))

    assert plan["truncated"] == []
    assert set(plan["chunked"]) == {
        a for members in plan["accessions"].values() for a in members
        if lengths[a] > 1024
    }


def test_the_plan_separates_residues_stored_from_residues_computed():
    """Overlapping windows are merged into one row per residue, so storage is the
    protein's length while compute is the sum of the windows. Reporting one as the
    other understates the forward pass by the overlap fraction."""
    plan = plan_probe(_pool(), ProbeSpec(per_band=5, chunk_size=512, chunk_overlap=128))

    assert plan["residues_computed"] > plan["residues_stored"]


def test_the_plan_costs_nothing_when_it_is_rejected():
    """Selection and sizing happen before any model is touched, so a refused plan
    is free rather than an hour wasted."""
    with pytest.raises(ValueError):
        plan_probe(_pool(), ProbeSpec(per_band=40, layers=tuple(range(49)),
                                      max_bytes=1024))


def test_the_plan_carries_the_layers_it_was_sized_for():
    plan = plan_probe(_pool(), ProbeSpec(per_band=5, layers=(0, 10)))

    assert plan["layers"] == [0, 10]


# ------------------------------------------------------------------------ chunking

def test_the_windows_cover_the_whole_sequence():
    """Not a prefix. That is the difference between chunking and truncating."""
    spans = chunk_spans(2500, 1024, 128)

    assert spans[0][0] == 0
    assert spans[-1][1] == 2500


def test_consecutive_windows_overlap_by_the_requested_amount():
    spans = chunk_spans(3000, 1024, 128)

    assert spans[1][0] == 1024 - 128


def test_a_short_sequence_is_one_window():
    assert chunk_spans(400, 1024, 128) == [(0, 400)]


def test_an_overlap_reaching_the_chunk_size_is_refused():
    """It would never advance, or would produce one window per residue."""
    with pytest.raises(ValueError, match="never advances"):
        chunk_spans(1000, 512, 512)


def test_computed_residues_exceeds_the_length_by_the_overlap():
    spec = ProbeSpec(chunk_size=512, chunk_overlap=128)

    assert computed_residues(2000, spec) > 2000


def test_a_protein_shorter_than_a_window_computes_exactly_its_length():
    assert computed_residues(300, ProbeSpec(chunk_size=1024)) == 300


def test_the_span_convention_matches_the_production_one():
    """Mirrored from PROTEA's _compute_chunk_spans, semantics included, so the
    probe and production partition a long protein the same way."""
    length, size, overlap = 2600, 1024, 128
    step = size - overlap
    expected = []
    start = 0
    while start < length:
        expected.append((start, min(start + size, length)))
        start += step

    assert chunk_spans(length, size, overlap) == expected

