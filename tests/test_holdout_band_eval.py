"""Holdout-band evaluator: row alignment + cafaeval result parsing.

F-RERANK-UNIVERSAL.5a replaced the degenerate v226-v227 interim-delta probe
(9-16 GT rows -> null cafaeval) with ``_eval_holdout_band``, which scores the
held-out v220-v226 snapshot band (``stage.val``) directly. Two pieces carry
the fix and are exercised here:

1. ``_read_val_meta`` must return row-aligned ``protein_accession`` / ``label``
   / ``go_term_id`` / ``aspect`` / ``vote_count``. ``proteins``/``aspects`` are
   stored one-per-group and must be expanded with ``groups``; ``labels`` and
   ``go_terms`` are already row-aligned; ``vote_count`` is read row-aligned from
   the bucket parquets, whose concat order matches the sidecars.
2. ``_run_cafaeval``'s result parser must index cafaeval's ``dfs_best`` by
   metric-KIND and match the row by ``ns`` (the prior parser keyed by namespace
   and read a non-existent ``metric`` field, so every value came back ``None``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from protea_reranker_lab.splits import StagedSplit
from protea_reranker_lab.universal_runner import _run_cafaeval
from protea_reranker_lab.universal_train import _read_val_meta


class _Stage:
    """Minimal stand-in exposing the attributes ``_read_val_meta`` reads."""

    def __init__(self, val: StagedSplit, feature_cols: list[str]):
        self.val = val
        self.feature_cols = feature_cols


def _write_val_split(tmp: Path) -> tuple[StagedSplit, list, list, list, list, list]:
    """Materialise a synthetic val split: 3 groups of sizes 2, 3, 1 (6 rows)."""
    # Per-group identity (one entry per group).
    proteins_per_group = ["P0", "P1", "P2"]
    aspects_per_group = ["mfo", "bpo", "mfo"]
    group_sizes = [2, 3, 1]
    # Row-aligned columns (6 rows total, in group-concat order).
    rows_prot = ["P0", "P0", "P1", "P1", "P1", "P2"]
    rows_asp = ["mfo", "mfo", "bpo", "bpo", "bpo", "mfo"]
    rows_label = [1, 0, 1, 0, 1, 0]
    rows_go = [f"GO:{i:07d}" for i in range(6)]
    rows_vote = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]

    out_dir = tmp / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Buckets carry ONLY feature columns. Split into two buckets to prove the
    # row-aligned concat order is preserved across bucket files.
    bucket_paths = []
    for bi, (lo, hi) in enumerate([(0, 4), (4, 6)]):
        bp = out_dir / f"bucket_{bi}.parquet"
        pq.write_table(
            pa.table({"vote_count": pa.array(rows_vote[lo:hi], pa.float64())}),
            str(bp),
        )
        bucket_paths.append(bp)

    labels_path = out_dir / "labels.npy"
    groups_path = out_dir / "groups.npy"
    proteins_path = out_dir / "proteins.npy"
    go_terms_path = out_dir / "go_terms.npy"
    aspects_path = out_dir / "aspects.npy"
    np.save(labels_path, np.array(rows_label, dtype=np.int8))
    np.save(groups_path, np.array(group_sizes, dtype=np.int32))
    np.save(proteins_path, np.array(proteins_per_group, dtype=object))
    np.save(go_terms_path, np.array(rows_go, dtype=object))
    np.save(aspects_path, np.array(aspects_per_group, dtype=object))

    split = StagedSplit(
        bucket_paths=bucket_paths,
        labels_path=labels_path,
        groups_path=groups_path,
        proteins_path=proteins_path,
        n_rows=6,
        n_groups=3,
        go_terms_path=go_terms_path,
        aspects_path=aspects_path,
    )
    return split, rows_prot, rows_asp, rows_label, rows_go, rows_vote


def test_read_val_meta_row_alignment(tmp_path: Path) -> None:
    split, rows_prot, rows_asp, rows_label, rows_go, rows_vote = _write_val_split(
        tmp_path
    )
    stage = _Stage(split, feature_cols=["vote_count"])

    meta = _read_val_meta(stage)

    # Per-group proteins/aspects expanded by group sizes -> row-aligned.
    assert list(meta["protein_accession"]) == rows_prot
    assert list(meta["aspect"]) == rows_asp
    # Labels / go terms read row-aligned, unchanged.
    assert list(meta["label"]) == rows_label
    assert list(meta["go_term_id"]) == rows_go
    # vote_count concatenated across buckets in the same row order.
    assert np.allclose(meta["vote_count"], rows_vote)
    # Every column has one entry per row.
    n = split.n_rows
    assert all(len(meta[k]) == n for k in meta)


def test_read_val_meta_no_vote_count_column(tmp_path: Path) -> None:
    """``vote_count`` is omitted when it is not a staged feature column."""
    split, *_ = _write_val_split(tmp_path)
    stage = _Stage(split, feature_cols=["some_other_feature"])

    meta = _read_val_meta(stage)

    assert "vote_count" not in meta
    assert len(meta["label"]) == split.n_rows


def test_run_cafaeval_parses_dfs_best(tmp_path: Path, monkeypatch) -> None:
    """The parser reads ``col`` at the row whose ``ns`` matches the aspect.

    cafaeval's ``dfs_best`` is keyed by metric-kind; each value is the
    best-threshold row(s) carrying ALL metrics as columns. We synthesise that
    shape and assert the picked numbers are non-null and correct (the old
    parser, keyed by ns + a non-existent ``metric`` field, returned all None).
    """
    cell = "nk-mfo"  # ns -> molecular_function
    cell_dir = tmp_path / cell
    cell_dir.mkdir(parents=True)
    out_json = cell_dir / "cafaeval_out.json"
    dfs_best = {
        "f": [
            {"ns": "molecular_function", "f": 0.55, "f_micro": 0.40, "f_micro_w": 0.61},
            {"ns": "biological_process", "f": 0.11, "f_micro": 0.10, "f_micro_w": 0.12},
        ],
        "f_micro": [
            {"ns": "molecular_function", "f": 0.50, "f_micro": 0.42, "f_micro_w": 0.58},
        ],
        "f_micro_w": [
            {"ns": "molecular_function", "f": 0.48, "f_micro": 0.39, "f_micro_w": 0.63},
        ],
    }

    def fake_run(*args, **kwargs):
        out_json.write_text(json.dumps(dfs_best))
        return None

    monkeypatch.setattr(
        "protea_reranker_lab.universal_runner.subprocess.run", fake_run
    )

    res = _run_cafaeval(
        cell, tmp_path, Path("/x.obo"), Path("/x.ia"), Path("/usr/bin/python"),
    )

    # Each metric picked from its own best-kind table, MFO row.
    assert res["fmax"] == 0.55
    assert res["f_micro"] == 0.42
    assert res["f_micro_w"] == 0.63
    assert all(v is not None for v in res.values())
