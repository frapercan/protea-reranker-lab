"""The database side of the reference pool, kept beside the bundle side.

``encoder_ablation`` can take its pool from a frozen bundle or from the database,
and the two must return the same four things or the ablation learns which side it
was run on. Holding them in one module rather than inside the ablation makes that
symmetry visible, and keeps the ablation module about the ablation.

This is the path that DRAWS the pool, so the ordered draw, the seeded sample and
the sequence-twin exclusion all live here. The bundle path does none of them,
because its producer already did, and doing them twice would describe a pool that
is not the one in use. See :mod:`protea_reranker_lab.gate_bundle`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _draw_reference_pool(cur: Any, spec: Any, queries: list[str], rng: Any) -> list[str]:
    """Exclude sequence twins, then draw the pool, in that order.

    Both steps carry a defect this code has already met once. The exclusion is by
    SEQUENCE and not by accession, because the bank is keyed by protein while
    embeddings key on sequence, so an accession merely sharing a sequence with a
    query arrived as a donor at cosine exactly 1.0, a guaranteed rank one, and at
    K=3 a third of the vote handed straight to the answer.

    And the ORDER BY is load-bearing rather than tidiness: without it the DISTINCT
    returns in whatever order the hash aggregate produces, which varies between
    runs, so a fixed seed shuffled a differently ordered list and selected a
    different pool every time.
    """
    qset = set(queries)
    cur.execute(
        """SELECT DISTINCT p2.accession
             FROM protein p1 JOIN protein p2 ON p2.sequence_id = p1.sequence_id
            WHERE p1.accession = ANY(%s)""",
        (list(queries),))
    twins = {a for (a,) in cur.fetchall()} - qset
    qset |= twins
    if twins:
        log.info("excluded %d accessions sharing a sequence with a query", len(twins))

    # ORDER BY is load-bearing, not tidiness. Without it Postgres returns this
    # DISTINCT in whatever order the hash aggregate produces, which varies between
    # runs, so shuffling with a fixed seed permuted the positions of a differently
    # ordered list and selected a different reference pool every time.
    cur.execute(
        """SELECT DISTINCT protein_accession FROM protein_go_annotation
           WHERE annotation_set_id = %s AND (hashtextextended(protein_accession, 42) %% %s) = 0
           ORDER BY protein_accession""",
        (spec.annotation_set_id, max(2, 556000 // (spec.ref_n * 2))))
    ref_accs = [a for (a,) in cur.fetchall() if a not in qset]
    if len(ref_accs) < spec.ref_n:
        # The modulo divisor is derived from ref_n, and at large ref_n the
        # integer division floors to zero and max() pins it at 2, capping the
        # candidate pool near half the corpus however many references are asked
        # for. Say so rather than silently training on fewer.
        log.warning("reference pool is %d, short of the requested ref_n=%d",
                    len(ref_accs), spec.ref_n)
    rng.shuffle(ref_accs)
    ref_accs = ref_accs[:spec.ref_n]
    rng.shuffle(ref_accs)
    return ref_accs[:spec.ref_n]


def load_pool_from_database(spec: Any, dag: Any, queries: list[str], rng: Any,
                            pull_mean: Any, propagate: Any) -> tuple:
    """Pull the t0 reference pool and the query embeddings, read-only.

    This is the path that draws the pool, so the draw and the twin exclusion both
    live here. The bundle path does neither, because its producer already did.
    """
    import psycopg2

    conn = psycopg2.connect(spec.dsn)
    cur = conn.cursor()
    # Exclude by SEQUENCE, not by accession. The bank is keyed by protein while
    # embeddings key on sequence_id, so an accession that merely shares a
    # sequence with a query survived an accession-level exclusion and arrived as
    # a donor at cosine exactly 1.0, which is a guaranteed rank 1. At K=3 that is
    # a third of the vote handed straight to the answer.
    ref_accs = _draw_reference_pool(cur, spec, queries, rng)
    ref_emb = pull_mean(cur, ref_accs, spec.embedding_config_id)
    q_emb = pull_mean(cur, queries, spec.embedding_config_id)
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id FROM protein_go_annotation pga
           JOIN go_term gt ON gt.id = pga.go_term_id
           WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)""",
        (spec.annotation_set_id, list(ref_emb.keys())))
    leaves: dict[str, list] = {}
    for a, go in cur.fetchall():
        leaves.setdefault(a, []).append(go)
    cur.close()
    conn.close()

    keep = [a for a in ref_emb if a in leaves and propagate(leaves[a], dag)]
    ref_clo = [propagate(leaves[a], dag) for a in keep]
    R = np.vstack([ref_emb[a] for a in keep]).astype(np.float32)
    q_accs = [a for a in queries if a in q_emb]
    Q = np.vstack([q_emb[a] for a in q_accs]).astype(np.float32)
    return R, Q, ref_clo, q_accs
