"""Running the mechanism grid over a bounded probe, and recording what it ran on.

The pieces exist separately: a variant space, a streaming applier, a functional
screen, and a stratified chunked probe. This assembles them and produces one table.

TWO DECISIONS STATED RATHER THAN BURIED

The atoms are the model's hidden dimensions. This first sweep applies k-WTA over
the layer's own dimensions rather than over a learned dictionary, which is exactly
what the recorded layer ablation did when it screened kWTA at 64, 128 and 256 and
found that standardisation rescued sparse. A learned dictionary is a separate axis
and a separate cost; running it first would confound "does the order matter" with
"does the encoder help", and the second question already has a large measured
answer while the first has none.

The probe's provenance is hashed. The target FASTA lives outside any repository,
so a rerun months later cannot otherwise tell whether it used the same sequences.
The digest travels in the result, which is the same reasoning as the target
export's content address.

HOW LAYERS ENTER THE CODE

A ``shared`` dictionary treats each (residue, layer) as one unit contributing to
the same atoms, so layer identity is marginalised away and an atom activated by
several layers is simply activated more. A ``partitioned`` dictionary gives each
layer its own block, so a residue is one unit in a wider space and layer identity
survives into the code. They are different representations of the same tensor and
the grid asks which one retrieval prefers.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from protea_reranker_lab.functional_proxy import (
    neighbour_purity,
    rank_agreement,
    sample_neighbour_pairs,
)
from protea_reranker_lab.mechanism import MechanismSpec, is_streamable
from protea_reranker_lab.mechanism_apply import apply_mechanism
from protea_reranker_lab.probe_store import ProbeStore, streaming_standardisation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepInputs:
    """Everything one sweep consumes, so a result can name its own inputs.

    The residues are a memory-mapped store rather than a dict of arrays. Holding
    them resident, and copying them once per variant to select layers, is what ran
    this machine out of memory twice: the mechanism was a streaming reduction and
    the harness around it was not.
    """

    store: ProbeStore
    closures: dict[str, frozenset[str]]
    source_digest: str

    @property
    def layers(self) -> tuple[int, ...]:
        return self.store.layers


def available_bytes() -> int:
    """Memory actually free, from the kernel rather than from total minus a guess."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def refuse_without_headroom(needed: int, *, reserve: int = 8 * 1024 ** 3) -> None:
    """Stop before starting if the machine cannot spare the working set.

    ``reserve`` is not padding. This machine is a compute node for another
    machine's grid, and its prediction worker has been measured at 12 GB while
    consuming batches. Exploration here has to leave that room, and the two
    out-of-memory kills that motivated this check happened because a probe was
    sized against total RAM as though nothing else lived here.
    """
    free = available_bytes()
    if free >= needed + reserve:
        return
    raise ValueError(
        f"this sweep needs about {needed / 1024 ** 3:.2f} GB and {free / 1024 ** 3:.2f} GB "
        f"is available, leaving less than the {reserve / 1024 ** 3:.0f} GB reserved for "
        "the workers this node runs for the other machine. Reduce the probe or wait "
        "until the grid is idle"
    )


def digest_of(path: str | Path) -> str:
    """Content address of the probe's source, so a rerun can prove it matched."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def parse_fasta(path: str | Path) -> dict[str, str]:
    """Accession to sequence, taking the field between the first two pipes if present.

    Mirrors the LAFA reader's rule so the probe and the submission agree on what an
    accession is, rather than agreeing by luck on the files we happen to hold.
    """
    sequences: dict[str, str] = {}
    accession, parts = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if accession:
                sequences[accession] = "".join(parts)
            header = line[1:].split()[0]
            fields = header.split("|")
            accession = fields[1] if len(fields) >= 2 else header
            parts = []
        elif line.strip():
            parts.append(line.strip())
    if accession:
        sequences[accession] = "".join(parts)
    return sequences


def flatten_layers(residues: np.ndarray, spec: MechanismSpec) -> np.ndarray:
    """(length, layers, width) into the (units, dictionary) the mechanism consumes.

    Shared marginalises layer identity by making each (residue, layer) a unit;
    partitioned preserves it by widening the dictionary. Nothing is averaged here,
    because averaging layers before the mechanism sees them would decide the
    question the grid is asking.
    """
    length, n_layers, width = residues.shape
    if spec.layer_mode == "partitioned" and n_layers > 1:
        return residues.reshape(length, n_layers * width)
    return residues.reshape(length * n_layers, width)


_STATS_CACHE: dict[tuple, tuple] = {}


def layer_index_for(inputs: SweepInputs, spec: MechanismSpec) -> list[int]:
    """Positions of the spec's layers inside the store's layer axis."""
    order = {layer: i for i, layer in enumerate(inputs.store.layers)}
    missing = [x for x in spec.layers if x not in order]
    if missing:
        raise ValueError(
            f"variant {spec.name} asks for layers {missing} and the probe holds "
            f"{list(inputs.store.layers)}; extract them or drop the variant rather "
            "than scoring it against a different representation"
        )
    return [order[x] for x in spec.layers]


def standardisation_of(inputs: SweepInputs, spec: MechanismSpec):
    """Corpus statistics for the z-score arm, fitted over the probe once.

    Fitted over the pooled probe rather than per protein: statistics computed
    within a protein make the representation depend on that protein's own length
    and composition, and the measured result that standardisation rescues sparse
    was obtained with pooled statistics.
    """
    if spec.normalize != "zscore":
        return None, None
    # Cached on what the statistics depend on, and accumulated protein by protein.
    # The previous version concatenated the whole probe, holding a second copy of
    # it for the duration, once per z-score arm.
    key = (inputs.source_digest, tuple(spec.layers), spec.layer_mode)
    if key not in _STATS_CACHE:
        _STATS_CACHE[key] = streaming_standardisation(
            inputs.store, layer_index_for(inputs, spec),
            lambda residues: flatten_layers(residues, spec),
        )
    return _STATS_CACHE[key]


def codes_for(inputs: SweepInputs, spec: MechanismSpec) -> tuple[np.ndarray, list[str]]:
    """One sequence code per protein, in accession order."""
    mean, std = standardisation_of(inputs, spec)
    index = layer_index_for(inputs, spec)
    codes, accessions = [], []
    for accession, residues in inputs.store.stream(index):
        codes.append(apply_mechanism(flatten_layers(residues, spec), spec,
                                     mean=mean, std=std))
        accessions.append(accession)
    return np.vstack(codes), accessions


def score_variant(inputs: SweepInputs, spec: MechanismSpec, *,
                  pairs: np.ndarray | None = None, purity_k: int = 10) -> dict:
    """Both screen measures for one variant, or the reason it could not be scored.

    A variant that cannot be screened is recorded with its reason rather than
    dropped, because a missing row and a bad row look the same in a table and only
    one of them is informative.
    """
    ok, reason = is_streamable(spec)
    if not ok:
        return {"variant": spec.name, "scored": False, "reason": reason}

    codes, accessions = codes_for(inputs, spec)
    closures = [inputs.closures[a] for a in accessions]
    result: dict = {"variant": spec.name, "scored": True,
                    "nonzero": float(np.mean((codes != 0).sum(axis=1)))}
    try:
        result.update(rank_agreement(codes, closures, pairs))
    except ValueError as exc:
        result.update({"spearman": None, "rank_reason": str(exc)[:120]})
    purity = neighbour_purity(codes, closures, purity_k)
    result["per_protein"] = purity.pop("per_protein")
    result.update(purity)
    result["accessions"] = accessions
    return result


def run_sweep(inputs: SweepInputs, specs: list[MechanismSpec], *,
              n_pairs: int = 20000, seed: int = 42) -> list[dict]:
    """Every variant against one fixed pair sample, so the comparison is paired."""
    rng = np.random.default_rng(seed)
    accessions = list(inputs.store.accessions)
    closures = [inputs.closures[a] for a in accessions]
    pairs = sample_neighbour_pairs(closures, n_pairs, rng)
    peak = inputs.store.peak_protein_bytes()
    refuse_without_headroom(peak * 4)
    log.info("sweeping %d variants over %d proteins and %d pairs, source %s, "
             "peak resident about %.0f MB",
             len(specs), len(accessions), pairs.shape[0], inputs.source_digest,
             peak / 1024 ** 2)

    rows = []
    for i, spec in enumerate(specs, start=1):
        rows.append({**score_variant(inputs, spec, pairs=pairs),
                     "source_digest": inputs.source_digest})
        if i % 25 == 0 or i == len(specs):
            log.info("  %d/%d variants", i, len(specs))
    return rows
