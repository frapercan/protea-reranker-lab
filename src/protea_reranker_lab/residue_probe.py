"""Selecting and sizing a bounded per-residue, per-layer probe set.

The mechanism is refined on residues and then deployed as a streaming reduction
over the corpus, so the residue tensor only ever has to exist for a probe. This
picks that probe, stratified by length, and says what it will cost BEFORE anything
is extracted.

The sizing is not a convenience. Residues by layers by width in float32 grows fast
enough that a plausible-looking request is often impossible: six sampled layers of
a 768-wide model over a 1,000-residue protein is 18 MB for one protein, and all 49
layers of it is 150 MB. Discovering that after an hour of forward passes is the
same shape as every other failure this project has met, so ``estimate_bytes`` is
called first and ``refuse_an_oversized_probe`` stops a request that does not fit.

FLOAT32 IS NOT A PREFERENCE HERE

Middle layers carry massive activations: layer 38 of ankh-base was measured at a
maximum absolute value of 440,611, against a float16 maximum of 65,504. Storing
those in half precision overflows them to infinity and silently corrupts the
result, which happened once and produced a purity of 0.076, the value random noise
gives. The same ceiling is why the database's halfvec type cannot hold this either.

WHY STRATIFIED BY LENGTH RATHER THAN SAMPLED PROPORTIONALLY

The measured gain from a better layer and normalisation concentrated on LONG
proteins, +41 per cent against +27 overall, which is the dilution signature: mean
pooling loses most where there is most to pool. A proportional sample would put
most of its power where the effect is smallest. Equal weight per band buys power
where the mechanism is expected to act, at the cost of a sample that does not
describe the corpus, which is the right trade for refining a mechanism and the
wrong one for reporting a rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: The project's length bands, mirrored from ``protea.core.strata`` so the probe
#: and the board never partition the corpus two different ways.
LENGTH_BANDS: tuple[tuple[str, int, int], ...] = (
    ("<=512", 1, 512),
    ("512-1024", 513, 1024),
    ("1024-2048", 1025, 2048),
    (">2048", 2049, 10 ** 9),
)


@dataclass(frozen=True)
class ProbeSpec:
    """What to extract, and the ceiling it must fit under."""

    per_band: int = 50
    layers: tuple[int, ...] = (0, 10, 19, 29, 38, 48)
    width: int = 768
    max_length: int = 2048
    seed: int = 42
    #: Bytes the extracted tensors may occupy. A ceiling rather than a guess,
    #: because the failure it prevents is discovered an hour in.
    max_bytes: int = 8 * 1024 ** 3


def band_of(length: int) -> str:
    for name, low, high in LENGTH_BANDS:
        if low <= length <= high:
            return name
    return LENGTH_BANDS[-1][0]


def select_stratified_probe(lengths: dict[str, int], spec: ProbeSpec) -> dict[str, list[str]]:
    """Equal numbers per length band, drawn reproducibly, ordered before sampling.

    The ordering is load-bearing for the same reason it was in the reference draw:
    sampling an unordered collection with a fixed seed selects a different set on
    every run, and nothing downstream can tell.
    """
    rng = np.random.default_rng(spec.seed)
    grouped: dict[str, list[str]] = {name: [] for name, _, _ in LENGTH_BANDS}
    for accession in sorted(lengths):
        grouped[band_of(lengths[accession])].append(accession)

    chosen: dict[str, list[str]] = {}
    for name, members in grouped.items():
        if len(members) <= spec.per_band:
            chosen[name] = list(members)
            if len(members) < spec.per_band:
                log.warning("band %s holds %d proteins, short of the %d requested",
                            name, len(members), spec.per_band)
        else:
            picked = rng.choice(len(members), size=spec.per_band, replace=False)
            chosen[name] = sorted(members[i] for i in picked)
    return chosen


def estimate_bytes(lengths: dict[str, int], chosen: dict[str, list[str]],
                   spec: ProbeSpec) -> int:
    """Exact float32 footprint of the residue tensors, truncation included.

    Summed over the actual selected proteins rather than from a mean length, since
    pushing a summary through a nonlinear function is how a storage figure came out
    wrong twice already.
    """
    residues = sum(
        min(lengths[a], spec.max_length)
        for members in chosen.values() for a in members
    )
    return residues * len(spec.layers) * spec.width * 4


def refuse_an_oversized_probe(size: int, spec: ProbeSpec) -> None:
    """Stop before the forward passes rather than after them."""
    if size <= spec.max_bytes:
        return
    raise ValueError(
        f"this probe would hold {size / 1024 ** 3:.2f} GB of residue tensors, over "
        f"the {spec.max_bytes / 1024 ** 3:.2f} GB ceiling. Reduce per_band, sample "
        "fewer layers, or lower max_length: at float32 the cost is residues by "
        "layers by width, and float16 is not available because middle layers "
        "overflow it"
    )


def plan_probe(lengths: dict[str, int], spec: ProbeSpec) -> dict:
    """Select, size and report, in that order, before anything is loaded.

    Returns the plan rather than the tensors so the decision to run is separable
    from the running, and so a rejected plan costs nothing.
    """
    chosen = select_stratified_probe(lengths, spec)
    size = estimate_bytes(lengths, chosen, spec)
    refuse_an_oversized_probe(size, spec)
    per_band = {name: len(members) for name, members in chosen.items()}
    residues = sum(min(lengths[a], spec.max_length)
                   for m in chosen.values() for a in m)
    log.info("probe: %d proteins across %d bands, %d residues, %.2f GB float32",
             sum(per_band.values()), len(per_band), residues, size / 1024 ** 3)
    return {
        "accessions": chosen,
        "per_band": per_band,
        "residues": residues,
        "bytes": size,
        "layers": list(spec.layers),
        "truncated": sorted(a for m in chosen.values() for a in m
                            if lengths[a] > spec.max_length),
    }
