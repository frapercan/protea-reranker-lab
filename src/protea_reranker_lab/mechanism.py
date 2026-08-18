"""The mechanism variants, and which of them survive the streaming requirement.

The plan is to refine the sparse code on a bounded set of per-residue embeddings
and then compute it over the whole corpus WITHOUT materialising intermediates. The
second half is not a deployment detail, it is a constraint on the first half: a
mechanism that cannot be written as a streaming reduction over residues can be
measured on the probe set and can never be run on the corpus, so measuring it
would be measuring something we cannot have.

A mechanism is streamable when it can be computed by consuming residues in order,
carrying a state whose size does not grow with protein length, and emitting the
sequence code at the end. mean, sum and max all qualify: their state is one vector
of dictionary width. A learned aggregator qualifies when its weight for a residue
depends only on that residue's own code. Attention across residues does NOT, in a
naive form, because the softmax denominator needs the whole set, though the online
softmax trick recovers it at the cost of a second accumulator.

``is_streamable`` returns the verdict and the reason, so a variant that cannot go
to the corpus is labelled at enumeration time rather than discovered after it wins.

WHAT THE AXES ARE FOR

Every axis here exists because something measured in this project makes it a live
question rather than a knob:

* ``normalize`` because the historical sparse negative was substantially a
  NORMALISATION confound: raw k-WTA selects the massive-activation dimensions, and
  z-scoring first is what rescued sparse into competitiveness.
* ``weighting`` because the recorded negative for sparsify-first used a BINARY
  code, where the aggregate degenerates to a frequency histogram and no single
  intense residue can carry an atom. Magnitude against frequency is that contrast
  made into an axis.
* ``layers`` because the last layer is measurably not the best one, and mid layers
  carry more functional signal.
* ``layer_mode`` because a shared dictionary marginalises layer identity while a
  partitioned one preserves it, and those answer different questions.
* ``local_granularity`` and ``k_local`` because top-k and averaging do not commute,
  which is the order axis.
* ``gate`` because interaction is the second half of the factorial, and it costs
  support: the intersection of two k-sparse codes is expected to hold k squared
  over D atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

#: Aggregators whose state is one fixed-width vector, so protein length does not
#: change the memory a corpus pass needs.
STREAMING_AGGREGATORS = frozenset({"mean", "sum", "max", "learned-local"})

#: Aggregators needing the whole residue set before any output. Measurable on the
#: probe, not runnable on the corpus in one pass.
BLOCKING_AGGREGATORS = frozenset({"attention", "set-encoder"})


@dataclass(frozen=True)
class MechanismSpec:
    """One point in the mechanism space. Everything a corpus pass would need."""

    name: str
    # --- the order axis
    local_granularity: str = "none"      # none | residue | cell
    cell_size: int = 256
    k_local: int | None = None           # None means no local sparsification
    # --- composition
    aggregator: str = "mean"             # mean | sum | max | learned-local | attention
    weighting: str = "magnitude"         # magnitude | frequency
    # --- the sequence code
    dictionary: int = 2048
    k_sequence: int = 128
    normalize: str = "zscore"            # raw | zscore
    # --- layers
    layers: tuple[int, ...] = (-1,)
    layer_mode: str = "shared"           # shared | partitioned
    layer_budget: str = "global"         # global | per-layer
    # --- interaction
    gate: bool = False

    @property
    def sparsifies_locally(self) -> bool:
        return self.local_granularity != "none" and self.k_local is not None

    @property
    def effective_dictionary(self) -> int:
        """Atoms the sequence code lives in.

        A partitioned layer mode gives each layer its own block, so the code is
        wider by the number of layers and the collision arithmetic changes with it.
        """
        if self.layer_mode == "partitioned":
            return self.dictionary * len(self.layers)
        return self.dictionary


def is_streamable(spec: MechanismSpec) -> tuple[bool, str]:
    """Whether this variant can run over the corpus without storing intermediates.

    Returned as a verdict plus a reason rather than a bare bool, because the reason
    is what tells a reader whether the limitation is fundamental or a missing
    implementation, and those have different costs.
    """
    if spec.aggregator in BLOCKING_AGGREGATORS:
        return False, (
            f"{spec.aggregator!r} needs every residue before it can emit anything, "
            "so a corpus pass would have to hold the protein's full residue tensor. "
            "An online-softmax formulation would recover it with a second "
            "accumulator; until that exists this variant is probe-only"
        )
    if spec.aggregator not in STREAMING_AGGREGATORS:
        return False, f"unknown aggregator {spec.aggregator!r}"
    if spec.gate and not spec.sparsifies_locally:
        return False, (
            "a gate modulates local codes by the sequence code, so it needs the "
            "sequence code before the locals are consumed. Without local "
            "sparsification there are no local codes to gate, and with them it "
            "needs two passes: one to build the sequence code, one to gate"
        )
    return True, "one pass, state is one vector of dictionary width"


def expected_gated_support(spec: MechanismSpec) -> float:
    """Atoms a gated code is expected to keep, k squared over the effective D."""
    d = spec.effective_dictionary
    if not spec.gate or d <= 0:
        return float(spec.k_sequence)
    return (spec.k_sequence * spec.k_sequence) / d


@dataclass(frozen=True)
class VariantGrid:
    """The axes to sweep. A named object rather than ten arguments, so a run can
    record the grid it swept beside the results it produced."""

    granularities: tuple[str, ...] = ("none", "cell", "residue")
    aggregators: tuple[str, ...] = ("mean", "sum", "max", "learned-local", "attention")
    weightings: tuple[str, ...] = ("magnitude", "frequency")
    normalizations: tuple[str, ...] = ("raw", "zscore")
    layer_sets: tuple[tuple[int, ...], ...] = ((-1,), (10,), (0, 10, 19, 29, 38, 48))
    layer_modes: tuple[str, ...] = ("shared", "partitioned")
    k_locals: tuple[int, ...] = (32, 128)
    dictionary: int = 2048
    k_sequence: int = 128
    gates: tuple[bool, ...] = (False, True)
    #: Below this many expected atoms a gated arm measures an empty vector rather
    #: than the hypothesis, so it is dropped instead of run.
    min_gated_support: int = 16


def enumerate_variants(grid: VariantGrid | None = None) -> list[MechanismSpec]:
    """The grid, with the combinations that cannot mean anything removed.

    Three prunings, each because the cell is a duplicate or an impossibility rather
    than a variant:

    * without local sparsification there is no local code, so ``k_local`` and
      ``weighting`` have nothing to act on and one representative is kept
    * ``layer_mode`` is meaningless for a single layer, since a partition into one
      block is the shared case
    * a gated arm whose expected support falls under the floor is dropped, because
      it would report a flat result for a reason unrelated to the hypothesis
    """
    g = grid or VariantGrid()
    out: list[MechanismSpec] = []
    seen: set[tuple] = set()
    for gran, agg, weight, norm, layers, mode, kl, gate in product(
        g.granularities, g.aggregators, g.weightings, g.normalizations,
        g.layer_sets, g.layer_modes, g.k_locals, g.gates,
    ):
        local = gran != "none"
        if not local and (weight != "magnitude" or kl != g.k_locals[0]):
            continue
        if len(layers) == 1 and mode != "shared":
            continue
        spec = MechanismSpec(
            name="", local_granularity=gran, k_local=kl if local else None,
            aggregator=agg, weighting=weight, normalize=norm, layers=layers,
            layer_mode=mode, dictionary=g.dictionary, k_sequence=g.k_sequence,
            gate=gate,
        )
        if gate and expected_gated_support(spec) < g.min_gated_support:
            continue
        key = (gran, agg, weight, norm, layers, mode, spec.k_local, gate)
        if key in seen:
            continue
        seen.add(key)
        out.append(replace(spec, name=variant_name(spec)))
    return out


def variant_name(spec: MechanismSpec) -> str:
    """A name that reconstructs the variant, so a result never needs a lookup table."""
    parts = [spec.local_granularity if spec.sparsifies_locally else "seq"]
    if spec.sparsifies_locally:
        parts.append(f"kl{spec.k_local}")
        parts.append(spec.weighting[:4])
    parts.append(spec.aggregator)
    parts.append(spec.normalize)
    parts.append("L" + "-".join(str(x) for x in spec.layers))
    if len(spec.layers) > 1:
        parts.append(spec.layer_mode[:4])
    if spec.gate:
        parts.append("gate")
    return ".".join(parts)


def probe_only(specs: list[MechanismSpec]) -> list[MechanismSpec]:
    """The variants that can be measured but never deployed, named as a group.

    Kept rather than filtered at enumeration, because a probe-only variant is still
    worth measuring: it bounds what a streamable one could achieve, and a large gap
    is an argument for building the two-pass form rather than for abandoning it.
    """
    return [s for s in specs if not is_streamable(s)[0]]
