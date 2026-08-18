"""Read the frozen reference pool so the encoder gates can run where the card is.

The two gates that decide whether the learned representation programme has a
result at all are cheap in compute and expensive in access. Six fits for the
fit-and-index disjointness condition, forty for the objective battery, and both
need mean-pooled embeddings and propagated GO closures for a reference pool.

Those live in the database, the database lives on the other machine, and the
standing rule is that this machine is never pointed at it. So the gates could not
run where the graphics card is, and the card is where they belong.

PROTEA's ``export_gate_bundle`` operation breaks that by publishing one compressed
archive. This reads it and hands back exactly what the database path hands back,
so the ablation does not learn which side its data came from.

Three things this deliberately does NOT do, each because the producer already did
it and doing it twice would be worse than not doing it at all:

* it does not re-sample the pool. The bundle is an ordered draw followed by a
  seeded sample, decided once by the operation. Shuffling it again here would be
  a second draw with a different seed lineage, and the manifest would then
  describe a pool that is not the one in use.
* it does not re-exclude sequence twins. The operation excluded every accession
  sharing a sequence with a query, which is the exclusion that matters because
  the bank is keyed by protein while embeddings key on sequence. Instead this
  VERIFIES that the manifest records the exclusion, and refuses a bundle that
  does not, since a bundle from an older producer would otherwise pass silently
  and hand every query its own vector back at cosine 1.0.
* it does not propagate against a frozen closure. Propagation happens here,
  against the ontology the caller is already holding, because a closure frozen at
  export time would be computed against an OBO the consumer cannot see.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Manifest keys without which the bundle cannot be trusted. ``excluded_by_sequence``
#: is the load-bearing one: its absence means the producer predates the sequence
#: level exclusion, and every query would retrieve itself.
REQUIRED_MANIFEST_KEYS = (
    "embedding_config_id",
    "annotation_set_id",
    "excluded_by_sequence",
    "ref_n_written",
    "dim",
)


@dataclass(frozen=True)
class GateBundle:
    """One frozen pool. The five travel together and are never used apart."""

    reference_accessions: list[str]
    reference_embeddings: np.ndarray
    query_accessions: list[str]
    query_embeddings: np.ndarray
    reference_leaves: dict[str, list[str]]
    manifest: dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.reference_embeddings.shape[1])


def load_gate_bundle(path: str | Path) -> GateBundle:
    """Parse the archive and refuse one that cannot be trusted.

    The refusals are deliberate rather than defensive. A bundle that is merely
    wrong in shape fails loudly at the first matrix operation; a bundle whose
    provenance is unstated fails nowhere at all and produces a plausible number,
    which is the failure mode this project keeps meeting.
    """
    with np.load(Path(path), allow_pickle=True) as archive:
        manifest = json.loads(str(archive["manifest"]))
        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
        if missing:
            raise ValueError(
                f"gate bundle manifest is missing {missing}. A bundle without "
                "excluded_by_sequence predates the sequence-level exclusion, and "
                "every query would retrieve its own vector at cosine 1.0"
            )
        bundle = GateBundle(
            reference_accessions=[str(a) for a in archive["reference_accessions"]],
            reference_embeddings=archive["reference_embeddings"].astype(np.float32),
            query_accessions=[str(a) for a in archive["query_accessions"]],
            query_embeddings=archive["query_embeddings"].astype(np.float32),
            reference_leaves=json.loads(str(archive["reference_leaves"])),
            manifest=manifest,
        )

    _check_shapes(bundle)
    log.info(
        "gate bundle: %d references, %d queries, dim %d, %d accessions excluded "
        "for sharing a sequence with a query",
        len(bundle.reference_accessions),
        len(bundle.query_accessions),
        bundle.dim,
        manifest["excluded_by_sequence"],
    )
    return bundle


def _check_shapes(bundle: GateBundle) -> None:
    """Rows must match accessions and both matrices must share a width.

    Checked rather than assumed because a mismatch here is silent: numpy will
    happily index a shorter matrix with a longer accession list and every
    downstream row is then attributed to the wrong protein, which is the same
    shape as the join corruption that once rewrote 8.3 per cent of rows.
    """
    if len(bundle.reference_accessions) != bundle.reference_embeddings.shape[0]:
        raise ValueError(
            f"{len(bundle.reference_accessions)} reference accessions against "
            f"{bundle.reference_embeddings.shape[0]} rows"
        )
    if len(bundle.query_accessions) != bundle.query_embeddings.shape[0]:
        raise ValueError(
            f"{len(bundle.query_accessions)} query accessions against "
            f"{bundle.query_embeddings.shape[0]} rows"
        )
    if bundle.reference_embeddings.shape[1] != bundle.query_embeddings.shape[1]:
        raise ValueError(
            f"references are {bundle.reference_embeddings.shape[1]}-dimensional "
            f"and queries are {bundle.query_embeddings.shape[1]}"
        )


def bundle_arm_data(bundle: GateBundle, dag: Any, propagate: Any) -> tuple:
    """Return exactly what the database loader returns, in the same order.

    ``(R, Q, ref_clo, q_accs)``. The ablation must not be able to tell which side
    its pool came from, so the contract is copied rather than adapted: references
    without a propagable closure are dropped here as they are there, and the count
    is reported rather than left for someone to notice a smaller matrix.
    """
    leaves = bundle.reference_leaves
    by_accession = dict(zip(bundle.reference_accessions, bundle.reference_embeddings, strict=True))

    keep = [a for a in bundle.reference_accessions if a in leaves and propagate(leaves[a], dag)]
    dropped = len(bundle.reference_accessions) - len(keep)
    if dropped:
        log.info("%d of %d references have no propagable closure and are dropped",
                 dropped, len(bundle.reference_accessions))
    if not keep:
        raise ValueError(
            "no reference in the bundle has a propagable closure. The bundle's "
            "annotation set and the ontology being propagated against are "
            "probably not the pair the producer recorded"
        )

    ref_clo = [propagate(leaves[a], dag) for a in keep]
    R = np.vstack([by_accession[a] for a in keep]).astype(np.float32)
    Q = bundle.query_embeddings.astype(np.float32)
    return R, Q, ref_clo, list(bundle.query_accessions)


def describe(bundle: GateBundle) -> str:
    """One line for a run log, so a result can name the pool that produced it."""
    m = bundle.manifest
    return (
        f"bundle config={m['embedding_config_id']} annotations={m['annotation_set_id']} "
        f"refs={len(bundle.reference_accessions)}/{m['ref_n_written']} "
        f"queries={len(bundle.query_accessions)} dim={bundle.dim} "
        f"twins_excluded={m['excluded_by_sequence']}"
    )


def load_pool_from_bundle(bundle_path, dag: Any, propagate: Any,
                          queries: list[str]) -> tuple:
    """The frozen pool, in the shape the database path returns.

    ``queries`` is not honoured here and that is deliberate. The producer computed
    the sequence-twin exclusion against the bundle's own query set, so scoring a
    different list would reintroduce exactly the donors that exclusion removed.
    A divergence is reported rather than silently reconciled, because a query set
    that quietly shrank is a population nobody can size afterwards.
    """
    bundle = load_gate_bundle(bundle_path)
    log.info("%s", describe(bundle))
    absent = set(queries) - set(bundle.query_accessions)
    if absent:
        log.warning(
            "%d requested queries are absent from the bundle and will not be "
            "scored; the bundle's query set is authoritative because the twin "
            "exclusion was computed against it",
            len(absent),
        )
    return bundle_arm_data(bundle, dag, propagate)
