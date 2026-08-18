"""Merging windows, and the alignment that would fail silently if it were wrong.

The extraction itself needs a card and a 5 GB checkpoint, so what is covered here
is everything around the forward pass: that overlapping windows tile the protein
and merge to one row per residue, that a tokenisation which is not one token per
residue is refused rather than shifting every position by one, and that an archive
whose offsets disagree with its matrix stops instead of handing each protein the
next one's residues.

The dtype is not covered by a unit test and is covered by measurement: on
ankh-base the per-layer maxima are 55 at layer 0, 55,697 at 10, 63,825 at 19,
152,788 at 29, 565,885 at 38 and 1 at 48, against a float16 ceiling of 65,504. Two
of those overflow, one sits at 97 per cent of the ceiling, and which layers are
safe therefore depends on the protein.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.residue_extract import (
    NONSTANDARD,
    ExtractionResult,
    extract_protein,
    load_probe,
    save_probe,
)
from protea_reranker_lab.residue_probe import ProbeSpec


class _StubModel:
    """Returns a window whose values encode the absolute residue position.

    Position-encoding on purpose: it makes a misaligned merge visible as a wrong
    value rather than as a plausible one.
    """

    def __init__(self, width=4, layers=51):
        self.width, self.layers = width, layers
        self.calls: list[int] = []

    def __call__(self, sequence: str, spec: ProbeSpec, offset: int) -> np.ndarray:
        self.calls.append(len(sequence))
        block = np.zeros((len(sequence), len(spec.layers), self.width), dtype=np.float32)
        for i in range(len(sequence)):
            block[i, :, :] = offset + i
        return block


# --------------------------------------------------------------------------- merging

def test_windows_tile_the_protein_and_merge_to_one_row_per_residue(monkeypatch):
    spec = ProbeSpec(layers=(0, 1), width=4, chunk_size=100, chunk_overlap=20)
    stub = _StubModel(width=4)
    monkeypatch.setattr(
        "protea_reranker_lab.residue_extract._window_layers",
        lambda tok, model, window, layers: stub(window, spec, 0),
    )

    got = extract_protein(None, None, "A" * 250, spec)

    assert got.residues.shape == (250, 2, 4)
    assert got.windows == len(stub.calls)


def test_every_residue_is_covered_by_at_least_one_window(monkeypatch):
    """A gap would leave zeros that look like a real representation."""
    spec = ProbeSpec(layers=(0,), width=2, chunk_size=64, chunk_overlap=8)
    monkeypatch.setattr(
        "protea_reranker_lab.residue_extract._window_layers",
        lambda tok, model, window, layers: np.ones(
            (len(window), len(spec.layers), 2), dtype=np.float32),
    )

    got = extract_protein(None, None, "A" * 200, spec)

    assert np.all(got.residues == 1.0)


def test_a_non_finite_extraction_is_refused(monkeypatch):
    """What half precision looks like after the fact, caught rather than stored."""
    spec = ProbeSpec(layers=(0,), width=2, chunk_size=64, chunk_overlap=8)
    monkeypatch.setattr(
        "protea_reranker_lab.residue_extract._window_layers",
        lambda tok, model, window, layers: np.full(
            (len(window), 1, 2), np.inf, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="non-finite"):
        extract_protein(None, None, "A" * 100, spec)


def test_a_short_protein_needs_one_window(monkeypatch):
    spec = ProbeSpec(layers=(0,), width=2, chunk_size=1024, chunk_overlap=128)
    monkeypatch.setattr(
        "protea_reranker_lab.residue_extract._window_layers",
        lambda tok, model, window, layers: np.ones(
            (len(window), 1, 2), dtype=np.float32),
    )

    assert extract_protein(None, None, "A" * 300, spec).windows == 1


# --------------------------------------------------------------------- the vocabulary

def test_non_standard_residues_are_mapped_as_production_maps_them():
    """U, Z, O and B are not in the vocabulary; production replaces them with X and
    the probe must measure the same representation."""
    assert NONSTANDARD.sub("X", "MUZOBK") == "MXXXXK"


def test_standard_residues_are_untouched():
    assert NONSTANDARD.sub("X", "ACDEFGHIKLMNPQRSTVWY") == "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------------------------- the archive

def test_the_archive_round_trips(tmp_path):
    spec = ProbeSpec(layers=(0, 1), width=3)
    extracted = {
        "P1": ExtractionResult("P1", np.ones((5, 2, 3), dtype=np.float32), 1),
        "P2": ExtractionResult("P2", 2 * np.ones((7, 2, 3), dtype=np.float32), 2),
    }
    path = tmp_path / "probe.npz"

    save_probe(path, extracted, spec)
    back = load_probe(path)

    assert set(back) == {"P1", "P2"}
    assert back["P1"].shape == (5, 2, 3)
    assert np.all(back["P2"] == 2.0)


def test_an_inconsistent_archive_stops_rather_than_shifting_every_protein(tmp_path):
    """Offsets disagreeing with the matrix hands each protein the next one's
    residues, which is the join-corruption shape in a file."""
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        accessions=np.array(["P1", "P2"], dtype=object),
        lengths=np.array([5, 7]),
        offsets=np.array([0, 5, 99]),
        residues=np.ones((12, 2, 3), dtype=np.float32),
        layers=np.array([0, 1]),
        windows=np.array([1, 1]),
    )

    with pytest.raises(ValueError, match="archive is inconsistent"):
        load_probe(path)
