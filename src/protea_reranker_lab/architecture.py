"""The encoding architecture as one object, and the bit budget that makes arms comparable.

Every comparison so far has held the dictionary at 2048 and moved one thing. That
made binary look worse than 2-bit, which is true at fixed width and is the wrong
question: the budget is not atoms, it is BITS, and binary values buy a wider
dictionary for the same bytes. Since accidental overlap between two k-sparse codes
runs as k squared over D, widening D is exactly the lever the retrieval
measurements asked for, so a narrow float arm and a wide binary arm can cost the
same and behave completely differently.

``bytes_per_sequence`` and ``matched_width`` exist so an arm is either declared at
a budget or compared at one. An arm that quietly costs twice as much as the one it
beats has not beaten it.

TWO PROJECTIONS, NOT ONE

The architecture carries a residue projection and a protein projection, either of
which may be absent. That covers the shipped champion (no residue projection,
protein projection on the pooled vector), the local-only routes, and the
two-stage form where one map learns to encode a residue and another learns to
encode the protein from the aggregate of those codes.

Whether the two share a dictionary is a field rather than an assumption. A
residue atom may be a local motif and a protein atom a functional class, and
forcing one vocabulary on both is a claim nobody has stated out loud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Precisions an arm may store its values at, in bits.
VALUE_BITS = (16, 4, 2, 1)


@dataclass(frozen=True)
class Projection:
    """One learned map into a sparse code."""

    width: int                 # dictionary size D
    keep: int                  # atoms retained, k; keep >= width means dense
    value_bits: int = 16
    #: What the map is trained against. ``None`` means it is not trained at all,
    #: which is the fixed-basis arm where the model's own dimensions are the atoms.
    objective: str | None = "lin"

    @property
    def is_dense(self) -> bool:
        return self.keep >= self.width

    @property
    def index_bits(self) -> int:
        """Bits to name one atom. A dense code names none: position is the name."""
        return 0 if self.is_dense else max(1, math.ceil(math.log2(self.width)))

    @property
    def bits_per_vector(self) -> int:
        if self.is_dense:
            return self.width * self.value_bits
        return self.keep * (self.index_bits + self.value_bits)

    @property
    def collision(self) -> float:
        """Expected shared atoms between two independent codes, k squared over D.

        Nominal. Measured collision runs above this whenever atom usage is
        concentrated, which is why effective width is reported beside it.
        """
        return 0.0 if self.is_dense else (self.keep * self.keep) / self.width


@dataclass(frozen=True)
class Architecture:
    """A full route from residues to a protein code.

    ``residue`` is the projection applied before aggregation and ``protein`` the
    one applied after. Either may be None:

    * residue None, protein set    the shipped champion: pool, then project
    * residue set, protein None    project locally, aggregate, done
    * both set                     two learned maps with the aggregation between
    * both None                    the fixed basis, model dimensions as atoms
    """

    name: str
    residue: Projection | None = None
    protein: Projection | None = None
    aggregator: str = "mean"
    normalize: str = "zscore"
    normalize_at: str = "residue"     # residue | aggregate | code
    layers: tuple[int, ...] = (-1,)
    shared_dictionary: bool = False   # only meaningful when both projections exist
    joint_training: bool = True       # one forward and one loss, or two stages

    @property
    def stages(self) -> int:
        return int(self.residue is not None) + int(self.protein is not None)

    @property
    def output(self) -> Projection | None:
        """The projection that produces what retrieval actually indexes."""
        return self.protein or self.residue

    def bytes_per_sequence(self) -> float:
        """What one protein costs in the store, which is the only stage that persists.

        Residue codes are NOT stored: the mechanism is a streaming reduction and
        the residue tensor exists only inside the forward pass. So the cost is the
        output projection alone, and an architecture with two learned maps costs
        exactly what its second one costs.
        """
        out = self.output
        return 0.0 if out is None else out.bits_per_vector / 8.0

    def describe(self) -> str:
        parts = [f"{self.stages}-stage", self.aggregator, self.normalize]
        out = self.output
        if out is not None:
            kind = "dense" if out.is_dense else f"k{out.keep}"
            parts.append(f"D{out.width}.{kind}.{out.value_bits}b")
        if self.stages == 2:
            parts.append("shared" if self.shared_dictionary else "split")
            parts.append("joint" if self.joint_training else "staged")
        return ".".join(parts)


def matched_width(budget_bytes: float, keep: int, value_bits: int) -> int:
    """Widest dictionary a sparse arm can afford at a byte budget.

    The point of the whole bit-budget framing: at 512 bytes and 128 atoms, float16
    values leave 16 bits for the index and a dictionary of 65,536, while 1-bit
    values leave 31 and the dictionary could be astronomically wide before the
    index runs out. In practice the binding constraint becomes the number of atoms
    the training can actually populate, not the arithmetic, which is why this
    returns a ceiling rather than a recommendation.
    """
    bits = budget_bytes * 8
    index_bits = bits / keep - value_bits
    if index_bits < 1:
        raise ValueError(
            f"{budget_bytes:.0f} bytes cannot hold {keep} atoms at {value_bits}-bit "
            f"values: the values alone need {keep * value_bits / 8:.0f} bytes and "
            "leave nothing to name them with"
        )
    return int(2 ** min(index_bits, 24))


def equal_budget_arms(budget_bytes: float, keep: int) -> list[tuple[int, int, float]]:
    """One arm per precision at a fixed byte budget, as (bits, width, collision).

    This is the comparison the earlier precision sweep could not make. Holding the
    dictionary at 2048 asked which precision loses least; holding the BYTES asks
    which spends them best, and those have different answers whenever collision is
    the binding constraint.
    """
    out = []
    for bits in VALUE_BITS:
        try:
            width = matched_width(budget_bytes, keep, bits)
        except ValueError:
            continue
        out.append((bits, width, (keep * keep) / width))
    return out
