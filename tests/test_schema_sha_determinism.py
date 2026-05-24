"""Determinism of the feature-schema sha computation.

The reranker cache and the booster reload path both pin a 12-hex
``schema_sha`` derived from the column set. If that sha is order- or
iteration-sensitive, the same dataset rebuilt in two different processes
will produce two different shas, silently invalidating every cached
artefact and breaking cross-PR comparability of the Ch6 grid.

These tests pin:

1. Repeated calls in the same process return the same sha (sanity).
2. Shuffling the column list before hashing returns the same sha
   (must be order-independent by spec; ``compute_schema_sha`` sorts).
3. Hashing the column list before AND after a dict -> ordered-dict
   roundtrip returns the same sha (catches dict-iteration leakage,
   stable on CPython 3.7+ but worth pinning at the boundary).
4. The sha for the canonical ``ALL_FEATURES`` set is stable across
   the parquet column-list path (post-parquet-roundtrip ordering),
   matching the direct compute.
"""

from __future__ import annotations

import io
import random

import pyarrow as pa
import pyarrow.parquet as pq

from protea_contracts import ALL_FEATURES, compute_schema_sha


def test_compute_schema_sha_repeated_calls_match() -> None:
    cols = list(ALL_FEATURES)
    a = compute_schema_sha(cols)
    b = compute_schema_sha(cols)
    assert a == b
    assert len(a) == 12


def test_compute_schema_sha_order_independent() -> None:
    base = list(ALL_FEATURES)
    shuffled = list(ALL_FEATURES)
    rng = random.Random(0)
    rng.shuffle(shuffled)
    assert shuffled != base, "shuffle was a no-op; pick a different seed"
    assert compute_schema_sha(base) == compute_schema_sha(shuffled)


def test_compute_schema_sha_after_dict_roundtrip() -> None:
    """A column list reconstructed from a dict's keys must hash identically.

    Catches the case where a producer materialises columns as a dict
    (e.g. ``{c: dtype for c in cols}``) and a consumer reads back
    ``list(d.keys())``. Insertion order is preserved on CPython 3.7+
    so this is a stability assertion, not a sort assertion.
    """
    cols = list(ALL_FEATURES)
    direct = compute_schema_sha(cols)
    as_dict = {c: i for i, c in enumerate(cols)}
    round_tripped = list(as_dict.keys())
    assert compute_schema_sha(round_tripped) == direct


def test_compute_schema_sha_after_parquet_roundtrip() -> None:
    """Hashing the column list of an in-memory parquet roundtrip must match.

    Failure mode this catches: pyarrow/parquet writers can reorder columns
    based on schema field iteration. Since ``compute_schema_sha`` sorts
    internally, the result must still be identical.
    """
    cols = list(ALL_FEATURES)
    schema = pa.schema([pa.field(c, pa.float32()) for c in cols])
    table = pa.Table.from_arrays(
        [pa.array([0.0], type=pa.float32()) for _ in cols],
        schema=schema,
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    read_back = pq.read_schema(buf)
    assert compute_schema_sha(read_back.names) == compute_schema_sha(cols)
