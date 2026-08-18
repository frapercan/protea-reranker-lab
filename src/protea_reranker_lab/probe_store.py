"""The probe on disk, read one protein at a time, never held whole.

Holding the probe in memory is what caused two out-of-memory kills on this
machine. The mechanism itself was written as a streaming reduction with a
fixed-width accumulator, and then the harness around it held every residue of
every protein, took a fresh copy per variant to select layers, and concatenated
the whole thing again to fit standardisation statistics. The principle was right
and it stopped at the function boundary.

So the probe is written once as a plain ``.npy`` and memory-mapped thereafter.
Plain rather than compressed on purpose: a compressed archive cannot be mapped, so
reading one protein from it means decompressing all of them, which is the very
thing being avoided.

Peak memory becomes one protein's residues rather than the corpus of them. At six
layers and 768 dimensions the largest protein in the probe is about 110 MB, against
9 GB for the whole probe, and the machine this runs on is also a compute node
serving another machine's grid, so the headroom is not ours to spend.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeStore:
    """A memory-mapped probe. Nothing here is resident except the index."""

    path: Path
    accessions: list[str]
    offsets: np.ndarray
    layers: tuple[int, ...]
    width: int

    @property
    def n_proteins(self) -> int:
        return len(self.accessions)

    @property
    def n_residues(self) -> int:
        return int(self.offsets[-1])

    def _matrix(self) -> np.ndarray:
        return np.load(self.path, mmap_mode="r")

    def residues(self, accession: str, layer_index: list[int] | None = None) -> np.ndarray:
        """One protein's residues, copied out of the map and nothing else.

        The layer selection happens HERE, on one protein, rather than over the
        whole probe. Selecting layers across the probe was a full duplicate of it
        per variant, which is the specific mistake that ran the machine out of
        memory.
        """
        i = self.accessions.index(accession)
        block = self._matrix()[int(self.offsets[i]):int(self.offsets[i + 1])]
        return np.asarray(block[:, layer_index, :] if layer_index else block)

    def stream(self, layer_index: list[int] | None = None) -> Iterator[tuple[str, np.ndarray]]:
        """Every protein in accession order, one resident at a time."""
        matrix = self._matrix()
        for i, accession in enumerate(self.accessions):
            block = matrix[int(self.offsets[i]):int(self.offsets[i + 1])]
            yield accession, np.asarray(block[:, layer_index, :] if layer_index else block)

    def peak_protein_bytes(self, n_layers: int | None = None) -> int:
        """Largest single protein, which is the resident set this design promises."""
        longest = int(np.max(np.diff(self.offsets))) if self.n_proteins else 0
        return longest * (n_layers or len(self.layers)) * self.width * 4


def write_probe(path: str | Path, extracted, layers: tuple[int, ...], width: int) -> ProbeStore:
    """Write the matrix uncompressed and its index beside it.

    Proteins are written in accession order so the index and the matrix cannot
    disagree, and the index is a separate small file so opening the probe does not
    touch the large one.
    """
    path = Path(path)
    accessions = sorted(extracted)
    lengths = [int(extracted[a].residues.shape[0]) for a in accessions]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    total = int(offsets[-1])
    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(total, len(layers), width)
    )
    for i, accession in enumerate(accessions):
        matrix[int(offsets[i]):int(offsets[i + 1])] = extracted[accession].residues
    matrix.flush()
    del matrix

    path.with_suffix(".index.json").write_text(json.dumps({
        "accessions": accessions, "offsets": offsets.tolist(),
        "layers": list(layers), "width": width,
    }))
    log.info("probe written: %d proteins, %d residues, %.2f GB, memory-mapped",
             len(accessions), total, total * len(layers) * width * 4 / 1024 ** 3)
    return open_probe(path)


def open_probe(path: str | Path) -> ProbeStore:
    """Open the index; the matrix is mapped lazily and never read whole."""
    path = Path(path)
    index = json.loads(path.with_suffix(".index.json").read_text())
    offsets = np.array(index["offsets"], dtype=np.int64)
    store = ProbeStore(path, index["accessions"], offsets,
                       tuple(index["layers"]), int(index["width"]))
    on_disk = np.load(path, mmap_mode="r")
    if on_disk.shape[0] != store.n_residues:
        raise ValueError(
            f"the index ends at {store.n_residues} residues and the matrix holds "
            f"{on_disk.shape[0]}; every protein after the first mismatch would read "
            "another's residues"
        )
    return store


def streaming_standardisation(store: ProbeStore, layer_index: list[int] | None,
                              flatten) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and standard deviation, accumulated protein by protein.

    Two passes of sums rather than one concatenation. The concatenation held a
    second copy of the whole probe for the duration, once per z-score arm, and the
    statistics it produced were identical to these.
    """
    total = count = None
    for _accession, residues in store.stream(layer_index):
        flat = flatten(residues)
        if total is None:
            total = flat.sum(axis=0, dtype=np.float64)
            square = (flat.astype(np.float64) ** 2).sum(axis=0)
            count = flat.shape[0]
        else:
            total += flat.sum(axis=0, dtype=np.float64)
            square += (flat.astype(np.float64) ** 2).sum(axis=0)
            count += flat.shape[0]
    if not count:
        raise ValueError("the probe is empty, so there are no statistics to fit")
    mean = total / count
    variance = np.maximum(square / count - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def extract_and_write(path, sequences: dict[str, str], spec, extract_one,
                      layers: tuple[int, ...], width: int) -> ProbeStore:
    """Extract and write protein by protein, never holding the probe in memory.

    The gap this closes: extracting into a dict and writing afterwards holds every
    residue of every protein at once, which is the 9 GB the memory-mapped store was
    built to avoid. The map is sized from the lengths, which are known before any
    forward pass, and each protein is written into its slot as it comes off the
    card.

    Peak resident becomes one protein plus the model, rather than the probe.
    """
    path = Path(path)
    accessions = sorted(sequences)
    lengths = [len(sequences[a]) for a in accessions]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total = int(offsets[-1])

    matrix = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(total, len(layers), width)
    )
    try:
        for i, accession in enumerate(accessions):
            block = extract_one(sequences[accession], spec)
            start, end = int(offsets[i]), int(offsets[i + 1])
            if block.shape[0] != end - start:
                raise ValueError(
                    f"{accession} is {end - start} residues and extraction returned "
                    f"{block.shape[0]}; writing it would shift every protein after it"
                )
            matrix[start:end] = block
            del block
            if (i + 1) % 25 == 0 or i + 1 == len(accessions):
                matrix.flush()
                log.info("  written %d/%d proteins", i + 1, len(accessions))
    finally:
        matrix.flush()
        del matrix

    path.with_suffix(".index.json").write_text(json.dumps({
        "accessions": accessions, "offsets": offsets.tolist(),
        "layers": list(layers), "width": width,
    }))
    log.info("probe written: %d proteins, %d residues, %.2f GB, memory-mapped",
             len(accessions), total, total * len(layers) * width * 4 / 1024 ** 3)
    return open_probe(path)

