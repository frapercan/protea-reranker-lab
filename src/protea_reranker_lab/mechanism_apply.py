"""Applying a mechanism, written as the streaming reduction it has to become.

The point of refining on a bounded probe set is to run the winner over the corpus
without materialising intermediates. That only holds if the mechanism is expressed
as a reduction in the first place, so this is written as one: residues arrive in
chunks, a fixed-width accumulator absorbs them, and the sequence code is emitted at
the end. Nothing here holds a protein's residue tensor.

Writing it this way rather than writing it convenient-first and porting later is
deliberate. A convenient implementation would let a non-streamable variant win on
the probe and only reveal the problem at corpus scale, which is the same shape as
every other defect this project has met: something that works, produces a number,
and cannot be had.

The accumulator is one vector of dictionary width per protein, plus a counter. For
max it is a running maximum; for mean and sum a running total. Length never enters
the state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from protea_reranker_lab.mechanism import MechanismSpec, is_streamable
from protea_reranker_lab.sparse_aggregation import topk_real


@dataclass
class Accumulator:
    """Fixed-width running state. Its size is the whole streaming claim."""

    total: np.ndarray
    count: int = 0
    mode: str = "mean"

    def absorb(self, block: np.ndarray) -> None:
        """Fold one chunk of local codes in. ``block`` is (units, dictionary)."""
        if self.mode == "max":
            self.total = np.maximum(self.total, block.max(axis=0))
        else:
            self.total += block.sum(axis=0)
        self.count += block.shape[0]

    def emit(self) -> np.ndarray:
        if self.mode == "mean" and self.count:
            return self.total / self.count
        return self.total


def standardize(X: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    """Per-dimension z-score, using statistics fitted elsewhere.

    Fitted elsewhere on purpose: statistics computed per protein would make the
    representation depend on the protein's own length and composition, and the
    measured result that standardisation rescues sparse was obtained with corpus
    statistics. Passing None leaves the input alone so ``raw`` is a real arm rather
    than a differently-scaled one.
    """
    if mean is None or std is None:
        return X
    return (X - mean) / np.where(std > 0, std, 1.0)


def local_codes(block: np.ndarray, spec: MechanismSpec) -> np.ndarray:
    """One chunk's local codes, sparsified and weighted as the spec asks.

    ``frequency`` is the binary reduction that the recorded negative used: every
    selected atom contributes one, so the aggregate becomes a usage count with no
    way for a single intense residue to carry an atom. It is kept as an arm rather
    than dismissed, because it is the control that says whether magnitude is what
    rescues the order.
    """
    if not spec.sparsifies_locally or spec.k_local is None:
        # sparsifies_locally already implies a k, but the contract types it as
        # optional, and a None reaching topk_real would select nothing silently
        # rather than raising.
        return block
    sparse = topk_real(block, spec.k_local)
    if spec.weighting == "frequency":
        return (sparse != 0).astype(np.float32)
    return sparse


def cells_of(residues: np.ndarray, spec: MechanismSpec):
    """Yield the units the mechanism aggregates over, in order, never all at once."""
    if spec.local_granularity == "cell":
        for start in range(0, residues.shape[0], spec.cell_size):
            yield residues[start:start + spec.cell_size].mean(axis=0, keepdims=True)
    elif spec.local_granularity == "residue":
        step = 512
        for start in range(0, residues.shape[0], step):
            yield residues[start:start + step]
    else:
        yield residues


def apply_mechanism(residues: np.ndarray, spec: MechanismSpec, *,
                    mean: np.ndarray | None = None,
                    std: np.ndarray | None = None) -> np.ndarray:
    """The sequence code, computed in one streaming pass over ``residues``.

    ``residues`` is (length, dictionary) already projected into code space. The
    projection itself is the encoder's job and is applied per chunk by the caller
    in a corpus run; here it is taken as given so the mechanism can be measured
    without a fitted encoder in the way.
    """
    ok, reason = is_streamable(spec)
    if not ok:
        raise ValueError(f"{spec.name} cannot be applied as a stream: {reason}")

    mode = "max" if spec.aggregator == "max" else (
        "sum" if spec.aggregator in {"sum", "learned-local"} else "mean")
    acc = Accumulator(total=np.zeros(residues.shape[1], dtype=np.float64), mode=mode)
    for block in cells_of(residues, spec):
        acc.absorb(local_codes(standardize(block, mean, std), spec))
    return topk_real(acc.emit()[None, :].astype(np.float32), spec.k_sequence)


def peak_state_bytes(spec: MechanismSpec, length: int) -> int:
    """Bytes the accumulator holds, which must not depend on ``length``.

    Exposed so the streaming claim is checkable rather than asserted, and so a
    future variant that quietly grows with protein length fails a test instead of a
    corpus run.
    """
    return spec.effective_dictionary * 8
