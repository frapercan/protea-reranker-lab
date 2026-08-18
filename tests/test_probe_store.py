"""Reading the probe one protein at a time, which is what stops it killing the box.

Two out-of-memory kills came from holding the probe resident: the mechanism was
written as a streaming reduction with a fixed-width accumulator, and the harness
around it held every residue of every protein, took a fresh copy per variant to
select layers, and concatenated the whole thing again to fit standardisation
statistics. The principle was right and it stopped at the function boundary.

Measured after the change on an 11.31 GB synthetic probe: peak ANONYMOUS memory
0.46 GB, which is 4 per cent of it, against a largest single protein of 70 MB. The
11.76 GB of total RSS is mapped file pages, which are evictable and are not what
the kernel kills for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from protea_reranker_lab.probe_store import (
    open_probe,
    streaming_standardisation,
    write_probe,
)


@dataclass
class _Extracted:
    residues: np.ndarray


def _probe(tmp_path, lengths=(5, 9, 3), layers=(0, 10), width=4, seed=0):
    rng = np.random.default_rng(seed)
    extracted = {
        f"P{i}": _Extracted(rng.random((n, len(layers), width)).astype(np.float32))
        for i, n in enumerate(lengths)
    }
    return write_probe(tmp_path / "probe.npy", extracted, layers, width), extracted


# ---------------------------------------------------------------------- round trip

def test_the_store_round_trips_every_protein(tmp_path):
    store, extracted = _probe(tmp_path)

    for accession, original in extracted.items():
        assert np.allclose(store.residues(accession), original.residues)


def test_proteins_are_stored_in_accession_order(tmp_path):
    """Index and matrix cannot disagree if both are built from one sorted list."""
    store, _ = _probe(tmp_path)

    assert store.accessions == sorted(store.accessions)


def test_the_matrix_is_uncompressed_so_it_can_be_mapped(tmp_path):
    """A compressed archive cannot be memory-mapped, so reading one protein would
    mean decompressing all of them, which is the thing being avoided."""
    store, _ = _probe(tmp_path)

    mapped = np.load(store.path, mmap_mode="r")

    assert isinstance(mapped, np.memmap)


def test_an_index_that_disagrees_with_the_matrix_is_refused(tmp_path):
    store, _ = _probe(tmp_path)
    index = json.loads(store.path.with_suffix(".index.json").read_text())
    index["offsets"][-1] += 100
    store.path.with_suffix(".index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match="another's residues"):
        open_probe(store.path)


# ------------------------------------------------------------------------ streaming

def test_streaming_yields_every_protein_once_in_order(tmp_path):
    store, _ = _probe(tmp_path)

    seen = [a for a, _ in store.stream()]

    assert seen == store.accessions


def test_layer_selection_happens_per_protein(tmp_path):
    """Selecting layers across the whole probe was a full duplicate of it per
    variant, which is the specific mistake that ran the machine out of memory."""
    store, extracted = _probe(tmp_path, layers=(0, 10, 19), width=4)

    for accession, block in store.stream([1]):
        assert block.shape[1] == 1
        assert np.allclose(block[:, 0, :], extracted[accession].residues[:, 1, :])


def test_the_peak_is_the_largest_protein_not_the_probe(tmp_path):
    store, _ = _probe(tmp_path, lengths=(5, 100, 3), layers=(0, 1), width=8)

    assert store.peak_protein_bytes() == 100 * 2 * 8 * 4


def test_the_peak_follows_the_selected_layer_count(tmp_path):
    store, _ = _probe(tmp_path, lengths=(10,), layers=(0, 1, 2), width=8)

    assert store.peak_protein_bytes(1) < store.peak_protein_bytes(3)


# ------------------------------------------------------------------ standardisation

def test_streaming_statistics_match_the_concatenated_ones(tmp_path):
    """The concatenation held a second copy of the whole probe for the duration,
    once per z-score arm, and produced exactly these numbers."""
    store, extracted = _probe(tmp_path, lengths=(7, 11, 5), layers=(0, 1), width=6)
    flatten = lambda r: r.reshape(r.shape[0] * r.shape[1], r.shape[2])

    mean, std = streaming_standardisation(store, None, flatten)
    stacked = np.concatenate([flatten(extracted[a].residues) for a in store.accessions])

    assert np.allclose(mean, stacked.mean(axis=0), atol=1e-5)
    assert np.allclose(std, stacked.std(axis=0), atol=1e-5)


def test_the_variance_never_goes_negative(tmp_path):
    """The sum-of-squares form can produce a tiny negative under cancellation, and
    a square root of it is nan, which would silently poison every z-scored arm."""
    store, _ = _probe(tmp_path, lengths=(4,), layers=(0,), width=3)
    flatten = lambda r: r.reshape(r.shape[0], r.shape[2])

    _mean, std = streaming_standardisation(store, None, flatten)

    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)
