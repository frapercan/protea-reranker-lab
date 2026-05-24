"""Roundtrip + shape guard on the v27-binary golden parquet fixture.

The thesis Ch6 champion (cafaeval Fmax 0.7291 +/- 0.0028) was trained on
the bench-v1-K5-v226-lineage-esm2_150m dataset under the v27-binary
recipe. A single-protein, single-cell slice of that dataset is checked
in at ``tests/fixtures/golden_v27.parquet`` so the lab can verify, in
CI, that a freshly built dataset still matches the column set, dtypes,
and manifest contract the champion was trained against.

If the fixture is missing on disk the test SKIPs with a clear pointer:
the parent agent or developer must drop a small parquet (one cell, one
aspect, <= a few thousand rows) at the canonical path. Generating it::

    poetry run python scripts/build_dataset.py \\
        --source-manifest <minio-manifest> \\
        --out-dir /tmp/golden_v27 \\
        --eval-snapshot-pair v226-v230 \\
        --train-snapshot-pairs v224-v226 \\
        --filter "category=NK and aspect=BPO" --max-rows 5000
    cp /tmp/golden_v27/eval.parquet tests/fixtures/golden_v27.parquet

What the test pins:

- Reserved columns ``RESERVED_COLUMNS`` are all present in the parquet.
- The numeric feature columns present match (a non-empty subset of)
  ``ALL_FEATURES`` from protea_contracts (parquet may legitimately drop
  PLM-specific columns when the relevant family is disabled; the test
  enforces that EVERY present feature column is one of ALL_FEATURES,
  i.e. no schema drift from rogue producers).
- Numeric feature columns have a float dtype (float32 or float64).
- No NaN/Inf in the ``label`` column; labels are in {0, 1}.
- Row count is positive (catches accidentally-empty fixtures).
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from protea_contracts import ALL_FEATURES, RESERVED_COLUMNS


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_v27.parquet"
DROP_HINT = (
    f"Golden parquet fixture missing at {FIXTURE_PATH}. "
    "Drop a small (<=5000 row) slice of "
    "bench-v1-K5-v226-lineage-esm2_150m there. "
    "See module docstring for the build command."
)


@pytest.fixture(scope="module")
def golden_table() -> pa.Table:
    if not FIXTURE_PATH.exists():
        pytest.skip(DROP_HINT)
    return pq.read_table(str(FIXTURE_PATH))


def test_fixture_exists_or_skip() -> None:
    """Explicit gate so the SKIP shows up by name in the report."""
    if not FIXTURE_PATH.exists():
        pytest.skip(DROP_HINT)
    assert FIXTURE_PATH.stat().st_size > 0


def test_reserved_columns_present(golden_table: pa.Table) -> None:
    names = set(golden_table.column_names)
    missing = [c for c in RESERVED_COLUMNS if c not in names]
    assert not missing, (
        f"Reserved columns missing from golden parquet: {missing}. "
        f"RESERVED_COLUMNS contract was violated by the producer."
    )


def test_feature_columns_are_subset_of_contract(golden_table: pa.Table) -> None:
    reserved = set(RESERVED_COLUMNS)
    contract = set(ALL_FEATURES)
    extras = [
        c
        for c in golden_table.column_names
        if c not in reserved and c not in contract
    ]
    assert not extras, (
        f"Golden parquet contains columns outside the contract: {extras}. "
        "Either add them to protea_contracts.ALL_FEATURES or remove them "
        "from the producer."
    )


def test_row_count_positive(golden_table: pa.Table) -> None:
    assert golden_table.num_rows > 0, "Golden parquet is empty."


def test_numeric_feature_dtypes(golden_table: pa.Table) -> None:
    """Numeric feature columns must be float (float32 or float64).

    Catches the case where a producer accidentally writes an integer
    dtype for a feature that was promoted to nullable float upstream.
    """
    reserved = set(RESERVED_COLUMNS)
    bad: list[tuple[str, str]] = []
    for field in golden_table.schema:
        if field.name in reserved:
            continue
        t = field.type
        if not (pa.types.is_floating(t) or pa.types.is_null(t)):
            bad.append((field.name, str(t)))
    assert not bad, (
        f"Non-float feature columns in golden parquet: {bad}. "
        "All feature columns must be float32/float64 (or null where the "
        "producer had no data)."
    )


def test_labels_binary_and_finite(golden_table: pa.Table) -> None:
    if "label" not in golden_table.column_names:
        pytest.skip("no label column in fixture (eval-only slice without labels)")
    labels = golden_table.column("label").to_pylist()
    bad_values = [
        v
        for v in labels
        if v is None
        or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        or v not in (0, 1)
    ]
    assert not bad_values[:5], (
        f"Found non-binary or non-finite label values (first 5): {bad_values[:5]}. "
        "Labels must be in {0, 1} with no NaN/Inf."
    )
