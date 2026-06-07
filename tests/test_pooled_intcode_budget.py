"""Tests for F-RERANK-UNIVERSAL.5a int-coding helpers and budget fix.

Acceptance criteria:
- _strings_to_int32 and _strings_to_int8 round-trip correctly.
- _compute_source_masks returns consistent masks on int-coded arrays.
- temporal val_strategy resolves holdout snapshot to its int code without
  treating code 0 as falsy (the bug that would cause an empty val split).
- _GLOBAL_ROW_BUDGET is calibrated to LightGBM training cost, not Pass-0
  string arrays.  Budget must be at most 70M rows (the old 80M was wrong).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from protea_reranker_lab.pooled_staging import (
    _GLOBAL_ROW_BUDGET,
    _compute_source_masks,
    _strings_to_int32,
    _strings_to_int8,
)
from protea_reranker_lab.staging import StagePlan


# ---------------------------------------------------------------------------
# Helper: write a minimal synthetic train parquet
# ---------------------------------------------------------------------------


def _write_train_parquet(
    path: Path,
    *,
    proteins: list[str],
    snapshot_pairs: list[str],
    labels: list[int],
    aspects: list[str],
) -> None:
    table = pa.table({
        "protein_accession": pa.array(proteins, pa.string()),
        "snapshot_pair": pa.array(snapshot_pairs, pa.string()),
        "label": pa.array(labels, pa.int8()),
        "aspect": pa.array(aspects, pa.string()),
    })
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Unit: int-coding helpers
# ---------------------------------------------------------------------------


class TestStringsToInt32:
    def test_round_trip(self) -> None:
        arr = np.array(["P12345", "Q99999", "P12345", "A00001"], dtype=object)
        codes, vocab = _strings_to_int32(arr)
        assert codes.dtype == np.int32
        assert len(vocab) == 3  # 3 distinct values
        assert list(vocab[codes]) == list(arr)

    def test_single_value(self) -> None:
        arr = np.array(["P00001"] * 100, dtype=object)
        codes, vocab = _strings_to_int32(arr)
        assert len(vocab) == 1
        assert np.all(codes == 0)

    def test_empty(self) -> None:
        arr = np.array([], dtype=object)
        codes, vocab = _strings_to_int32(arr)
        assert len(codes) == 0
        assert len(vocab) == 0

    def test_memory_dtype(self) -> None:
        """int32 codes use 4 bytes/row vs. 8-byte object pointers + heap strings."""
        arr = np.array(["UniProtAC_" + str(i % 1000) for i in range(10_000)], dtype=object)
        codes, _ = _strings_to_int32(arr)
        assert codes.dtype == np.int32
        assert codes.nbytes == len(arr) * 4  # 4 bytes/row


class TestStringsToInt8:
    def test_round_trip(self) -> None:
        arr = np.array(["MFO", "BPO", "CCO", "MFO", "BPO"], dtype=object)
        codes, vocab = _strings_to_int8(arr)
        assert codes.dtype == np.int8
        assert list(vocab[codes]) == list(arr)

    def test_too_many_values_raises(self) -> None:
        arr = np.array([str(i) for i in range(200)], dtype=object)
        with pytest.raises(ValueError, match="Too many distinct values"):
            _strings_to_int8(arr)


# ---------------------------------------------------------------------------
# Integration: _compute_source_masks
# ---------------------------------------------------------------------------


def _make_plan(**kwargs) -> StagePlan:
    defaults = dict(
        val_strategy="none",
        val_fraction=0.1,
        val_holdout_snapshot=None,
        neg_pos_ratio=5.0,
        seed=42,
        parent_map_path=None,
        bucket_count=4,
        carry_go_terms=False,
        batch_size=50_000,
        train_snapshot_pairs=None,
        eval_snapshot_pair=None,
    )
    defaults.update(kwargs)
    return StagePlan(**defaults)


class TestComputeSourceMasksIntCoded:
    def test_masks_shape_and_dtype(self, tmp_path: Path) -> None:
        """keep and val masks must be boolean arrays of the same length as the parquet."""
        n = 200
        proteins = ["P%05d" % (i % 20) for i in range(n)]
        pairs = ["2020-01" if i < 100 else "2021-01" for i in range(n)]
        labels = [1 if i % 5 == 0 else 0 for i in range(n)]
        aspects = ["MFO" if i % 3 == 0 else ("BPO" if i % 3 == 1 else "CCO") for i in range(n)]
        pq_path = tmp_path / "train.parquet"
        _write_train_parquet(pq_path, proteins=proteins, snapshot_pairs=pairs,
                             labels=labels, aspects=aspects)
        plan = _make_plan()
        masks = _compute_source_masks(pq_path, plan=plan)
        assert masks.keep.dtype == bool
        assert masks.val.dtype == bool
        assert len(masks.keep) == n
        assert len(masks.val) == n

    def test_temporal_strategy_correct_val_mask(self, tmp_path: Path) -> None:
        """temporal val_strategy must mark the holdout snapshot rows as val=True.

        This test catches the code-0-is-falsy bug: if 'goa:2020-01' sorts to
        code 0 after np.unique and the temporal branch does ``if not 0`` it
        would raise ValueError instead of building the val mask.
        """
        n = 100
        # 'goa:2020-01' sorts BEFORE 'goa:2021-01' lexicographically, so it
        # gets code=0 after _strings_to_int32 -- the dangerous case.
        holdout = "goa:2020-01"
        other = "goa:2021-01"
        proteins = ["P%05d" % (i % 10) for i in range(n)]
        pairs = [holdout if i < 40 else other for i in range(n)]
        labels = [1 if i % 4 == 0 else 0 for i in range(n)]
        aspects = ["MFO"] * n
        pq_path = tmp_path / "train.parquet"
        _write_train_parquet(pq_path, proteins=proteins, snapshot_pairs=pairs,
                             labels=labels, aspects=aspects)
        plan = _make_plan(val_strategy="temporal", val_holdout_snapshot=holdout)
        masks = _compute_source_masks(pq_path, plan=plan)
        # Val mask should be True exactly where pairs == holdout
        expected_val = np.array([p == holdout for p in pairs])
        np.testing.assert_array_equal(masks.val, expected_val)

    def test_temporal_strategy_absent_snapshot_gives_empty_val(self, tmp_path: Path) -> None:
        """If holdout snapshot is not in this source's pairs, val is all-False."""
        n = 60
        proteins = ["P%05d" % (i % 6) for i in range(n)]
        pairs = ["goa:2021-01"] * n
        labels = [0] * n
        aspects = ["BPO"] * n
        pq_path = tmp_path / "train.parquet"
        _write_train_parquet(pq_path, proteins=proteins, snapshot_pairs=pairs,
                             labels=labels, aspects=aspects)
        plan = _make_plan(val_strategy="temporal",
                          val_holdout_snapshot="goa:2020-01")  # absent
        masks = _compute_source_masks(pq_path, plan=plan)
        assert not masks.val.any(), "absent holdout must yield empty val split"

    def test_empty_source_returns_empty_masks(self, tmp_path: Path) -> None:
        """A source with 0 rows (after snapshot filter) must return empty masks."""
        pq_path = tmp_path / "train.parquet"
        _write_train_parquet(pq_path, proteins=[], snapshot_pairs=[],
                             labels=[], aspects=[])
        plan = _make_plan(train_snapshot_pairs=["goa:2020-01"])
        masks = _compute_source_masks(pq_path, plan=plan)
        assert len(masks.keep) == 0
        assert len(masks.val) == 0


# ---------------------------------------------------------------------------
# Global budget calibration
# ---------------------------------------------------------------------------


class TestGlobalRowBudget:
    def test_budget_is_lower_than_old_wrong_value(self) -> None:
        """Old budget of 80M was calibrated on ~30 B/row (string object arrays).

        The real LightGBM training cost for 56 features is ~520 B/row, giving a
        correct budget of ~60M rows for a 30 GB headroom on a 62 GB box.
        Budget must be at most 70M to catch a regression back to the wrong value.
        """
        assert _GLOBAL_ROW_BUDGET is not None
        assert _GLOBAL_ROW_BUDGET <= 70_000_000, (
            f"_GLOBAL_ROW_BUDGET={_GLOBAL_ROW_BUDGET} exceeds 70M; "
            "was it reset to the old miscalibrated value?"
        )

    def test_budget_is_positive_and_reasonable(self) -> None:
        """Budget must be large enough to be useful (at least 30M rows)."""
        assert _GLOBAL_ROW_BUDGET is not None
        assert _GLOBAL_ROW_BUDGET >= 30_000_000
