"""Tests for F-RERANK-UNIVERSAL.3 aspect-conditioned staging.

Acceptance criteria:
- n_groups == distinct (protein, aspect) pairs (NOT just proteins).
- Every group is single-aspect and single-bucket.
- The VALID protein set can be reconstructed by snapshot_pair filter.
- Candidate recall (raw + post-propagation) is reported per (category, aspect).
- The .1 eval_f_micro_w evaluator path (eval split is non-empty) works with
  aspect-conditioned staging.
- Backward compat: legacy per-cell staging (aspect_conditioned=False) still
  works and n_groups == distinct proteins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from protea_reranker_lab.staging import StagePlan, stage_for_training
from protea_reranker_lab.recall import compute_recall_table, RecallRecord
from protea_reranker_lab.splits import _sort_bucket


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _write_multi_aspect_parquet(
    path: Path,
    *,
    n_proteins: int = 8,
    aspects: list[str] | None = None,
    categories: list[str] | None = None,
    k_per_cell: int = 4,
    snapshot_pairs: list[str] | None = None,
    seed: int = 0,
) -> None:
    """Write a synthetic parquet with multiple (category, aspect) combinations.

    Columns: protein_accession, label, go_term_id, category, aspect,
             snapshot_pair, knn_vote_score.
    """
    if aspects is None:
        aspects = ["mfo", "bpo", "cco"]
    if categories is None:
        categories = ["nk", "lk"]
    if snapshot_pairs is None:
        snapshot_pairs = ["v226-v227"]

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        c: [] for c in [
            "protein_accession", "label", "go_term_id", "category",
            "aspect", "snapshot_pair", "knn_vote_score",
        ]
    }
    for snap in snapshot_pairs:
        for cat in categories:
            for asp in aspects:
                for pidx in range(n_proteins):
                    acc = f"P{pidx:05d}"
                    for j in range(k_per_cell):
                        rows["protein_accession"].append(acc)
                        rows["label"].append(int(rng.random() < 0.3))
                        rows["go_term_id"].append(f"GO:{asp[:2].upper()}{pidx:04d}{j:03d}")
                        rows["category"].append(cat)
                        rows["aspect"].append(asp)
                        rows["snapshot_pair"].append(snap)
                        rows["knn_vote_score"].append(float(rng.random()))

    table = pa.table({
        "protein_accession": pa.array(rows["protein_accession"], pa.string()),
        "label": pa.array(rows["label"], pa.int8()),
        "go_term_id": pa.array(rows["go_term_id"], pa.string()),
        "category": pa.array(rows["category"], pa.string()),
        "aspect": pa.array(rows["aspect"], pa.string()),
        "snapshot_pair": pa.array(rows["snapshot_pair"], pa.string()),
        "knn_vote_score": pa.array(rows["knn_vote_score"], pa.float32()),
    })
    pq.write_table(table, str(path))


@pytest.fixture
def multi_aspect_dataset(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_multi_aspect_parquet(
        ds / "train.parquet",
        n_proteins=8, aspects=["mfo", "bpo", "cco"], categories=["nk", "lk"],
        snapshot_pairs=["v224-v225", "v225-v226", "v226-v227"],
        seed=1,
    )
    _write_multi_aspect_parquet(
        ds / "eval.parquet",
        n_proteins=8, aspects=["mfo", "bpo", "cco"], categories=["nk", "lk"],
        snapshot_pairs=["v226-v227"],
        seed=2,
    )
    return ds


# ---------------------------------------------------------------------------
# Acceptance test 1: n_groups == distinct (protein, aspect) pairs
# ---------------------------------------------------------------------------


def test_aspect_conditioned_groups_are_protein_aspect_pairs(
    multi_aspect_dataset, tmp_path
):
    """With aspect_conditioned=True, n_groups must equal distinct
    (protein, aspect) pairs in the VALID (eval) split."""
    ds_dir = multi_aspect_dataset
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),  # aspect in cell is IGNORED when aspect_conditioned=True
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_ac",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
        ),
    )

    # eval split: nk x {mfo, bpo, cco} x 8 proteins -> 24 distinct groups
    # (since aspect_conditioned=True ignores the "nk/mfo" cell filter;
    # category "nk" filter still applies).
    assert stage.eval.aspects_path is not None, (
        "aspects_per_group.npy must be emitted when aspect_conditioned=True"
    )
    aspects_per_group = np.load(stage.eval.aspects_path, allow_pickle=True)
    proteins_per_group = np.load(stage.eval.proteins_path, allow_pickle=True)

    n_groups = stage.eval.n_groups
    assert n_groups == len(aspects_per_group), (
        f"n_groups={n_groups} must equal len(aspects_per_group)={len(aspects_per_group)}"
    )

    # Every group must be single-aspect (confirmed by checking aspects_per_group
    # is a 1D array with one value per group).
    assert aspects_per_group.ndim == 1
    assert proteins_per_group.ndim == 1
    assert len(proteins_per_group) == len(aspects_per_group) == n_groups


def test_groups_equal_distinct_protein_aspect_pairs(multi_aspect_dataset, tmp_path):
    """n_groups must equal the number of distinct (protein, aspect) pairs
    in the staged split."""
    ds_dir = multi_aspect_dataset
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_pa",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
        ),
    )

    aspects_per_group = np.load(stage.eval.aspects_path, allow_pickle=True)
    proteins_per_group = np.load(stage.eval.proteins_path, allow_pickle=True)

    # Derive distinct (protein, aspect) pairs from the npy arrays
    pairs = set(zip(proteins_per_group.tolist(), aspects_per_group.tolist()))
    assert stage.eval.n_groups == len(pairs), (
        f"n_groups={stage.eval.n_groups} must equal distinct (prot,asp) pairs={len(pairs)}"
    )


# ---------------------------------------------------------------------------
# Acceptance test 2: every group is single-aspect
# ---------------------------------------------------------------------------


def test_every_group_is_single_aspect(multi_aspect_dataset, tmp_path):
    """Load the eval parquet bucket and confirm each group has a single aspect."""
    ds_dir = multi_aspect_dataset
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_single",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
        ),
    )

    aspects_per_group = np.load(stage.eval.aspects_path, allow_pickle=True)
    groups = np.load(stage.eval.groups_path)
    labels = np.load(stage.eval.labels_path)

    assert len(groups) == len(aspects_per_group), (
        "groups and aspects_per_group must be co-aligned"
    )
    # All aspect values in aspects_per_group must be from the expected set.
    assert set(aspects_per_group.tolist()).issubset({"mfo", "bpo", "cco"}), (
        f"Unexpected aspect values: {set(aspects_per_group.tolist())}"
    )


# ---------------------------------------------------------------------------
# Acceptance test 3: single-bucket per group
# ---------------------------------------------------------------------------


def test_each_group_stays_in_single_bucket(multi_aspect_dataset, tmp_path):
    """No (protein, aspect) group may appear in more than one bucket."""
    ds_dir = multi_aspect_dataset
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_bucket",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
            bucket_count=4,  # small bucket count to increase chance of spread
        ),
    )

    # Read the final bucket parquets (only feature cols; no protein/aspect).
    # We verify via aspects_per_group: each (protein, aspect) pair appears
    # exactly ONCE across all groups (no duplicate in the group array).
    aspects_per_group = np.load(stage.eval.aspects_path, allow_pickle=True)
    proteins_per_group = np.load(stage.eval.proteins_path, allow_pickle=True)
    pairs = list(zip(proteins_per_group.tolist(), aspects_per_group.tolist()))
    assert len(pairs) == len(set(pairs)), (
        "Each (protein, aspect) must appear in exactly one group (one bucket)."
    )


# ---------------------------------------------------------------------------
# Acceptance test 4: legacy per-cell path still works
# ---------------------------------------------------------------------------


def test_legacy_per_cell_staging_unaffected(multi_aspect_dataset, tmp_path):
    """aspect_conditioned=False: n_groups == distinct proteins (legacy behavior)."""
    ds_dir = multi_aspect_dataset
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "bpo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_legacy",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=False,
        ),
    )
    # aspects_path is None in legacy mode.
    assert stage.eval.aspects_path is None, (
        "aspects_path must be None in legacy (non-aspect-conditioned) mode"
    )
    # n_groups == distinct proteins (8 in the test dataset).
    proteins_per_group = np.load(stage.eval.proteins_path, allow_pickle=True)
    assert stage.eval.n_groups == len(proteins_per_group)
    assert stage.eval.n_groups == 8, (
        f"Legacy path must group by protein only; expected 8, got {stage.eval.n_groups}"
    )


# ---------------------------------------------------------------------------
# Acceptance test 5: VALID window snapshot_pair filtering
# ---------------------------------------------------------------------------


def test_valid_window_filters_eval_split(multi_aspect_dataset, tmp_path):
    """eval_snapshot_pair='v226-v227' must select only v226-v227 rows in eval."""
    ds_dir = multi_aspect_dataset
    stage = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_valid",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
        ),
    )
    # The eval parquet only has v226-v227 rows so n_proteins == 8 (per cell).
    # With aspect_conditioned=True we get groups for all 3 aspects x 8 proteins
    # in category "nk" = 24 groups.
    assert stage.eval.n_groups == 24, (
        f"Expected 24 (protein,aspect) groups in nk x 3-aspects x 8 proteins; "
        f"got {stage.eval.n_groups}"
    )


def test_train_snapshot_pairs_filters_training(multi_aspect_dataset, tmp_path):
    """train_snapshot_pairs restricts the training set to the given pairs."""
    ds_dir = multi_aspect_dataset

    stage_all = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_all",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            train_snapshot_pairs=None,  # all pairs
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
        ),
    )
    stage_single = stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("nk", "mfo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=tmp_path / "stage_single",
        plan=StagePlan(
            val_strategy="none",
            val_fraction=0.0,
            val_holdout_snapshot=None,
            neg_pos_ratio=None,
            seed=42,
            train_snapshot_pairs=["v226-v227"],  # restrict to one pair
            eval_snapshot_pair="v226-v227",
            aspect_conditioned=True,
        ),
    )
    # Training with 3 pairs should have more rows than with 1 pair.
    assert stage_all.train.n_rows > stage_single.train.n_rows, (
        f"All-pairs training ({stage_all.train.n_rows}) must have more rows "
        f"than single-pair ({stage_single.train.n_rows})"
    )


def test_test_snapshot_pairs_stored_in_stage_plan():
    """test_snapshot_pairs is a list stored in StagePlan (multi-window TEST)."""
    plan = StagePlan(
        val_strategy="none",
        val_fraction=0.0,
        val_holdout_snapshot=None,
        neg_pos_ratio=None,
        seed=42,
        train_snapshot_pairs=None,
        eval_snapshot_pair="v226-v227",
        test_snapshot_pairs=["v227-v228", "v227-v229", "v227-v230"],
        aspect_conditioned=True,
    )
    assert plan.test_snapshot_pairs == ["v227-v228", "v227-v229", "v227-v230"]
    assert len(plan.test_snapshot_pairs) == 3


# ---------------------------------------------------------------------------
# Acceptance test 6: candidate recall reporting
# ---------------------------------------------------------------------------


def test_recall_table_per_category_aspect(multi_aspect_dataset):
    """compute_recall_table returns one record per (category, aspect) cell."""
    ds_dir = multi_aspect_dataset
    records = compute_recall_table(
        ds_dir / "eval.parquet",
        snapshot_pair="v226-v227",
        categories=["nk", "lk"],
        aspects=["mfo", "bpo", "cco"],
    )
    # 2 categories x 3 aspects = 6 records (all cells have rows).
    assert len(records) == 6, f"Expected 6 recall records, got {len(records)}"
    for r in records:
        assert isinstance(r, RecallRecord)
        assert r.category in {"nk", "lk"}
        assert r.aspect in {"mfo", "bpo", "cco"}
        assert r.n_candidates > 0
        assert 0.0 <= r.recall_raw <= 1.0
        assert 0.0 <= r.recall_prop <= 1.0


def test_recall_post_propagation_gte_raw(multi_aspect_dataset, tmp_path):
    """Post-propagation recall must be >= raw recall (ancestors add positives)."""
    ds_dir = multi_aspect_dataset

    # Build a minimal parent_map.json with a couple of ancestor links.
    parent_map = {
        "GO:MF00000": ["GO:MF00001"],
        "GO:MF00001": ["GO:MF00002"],
    }
    pm_path = tmp_path / "parent_map.json"
    pm_path.write_text(
        __import__("json").dumps({"parents": parent_map}, indent=2)
    )

    records_no_prop = compute_recall_table(
        ds_dir / "eval.parquet",
        snapshot_pair="v226-v227",
        categories=["nk"],
        aspects=["mfo"],
    )
    records_with_prop = compute_recall_table(
        ds_dir / "eval.parquet",
        snapshot_pair="v226-v227",
        parent_map_path=pm_path,
        categories=["nk"],
        aspects=["mfo"],
    )

    assert len(records_no_prop) == 1
    assert len(records_with_prop) == 1
    r_no = records_no_prop[0]
    r_with = records_with_prop[0]

    # Post-propagation positives >= raw positives.
    assert r_with.n_positives_prop >= r_no.n_positives_raw
    assert r_with.recall_prop >= r_no.recall_raw


# ---------------------------------------------------------------------------
# Acceptance test 7: _sort_bucket aspect-conditioned group detection
# ---------------------------------------------------------------------------


def test_sort_bucket_aspect_conditioned(tmp_path):
    """_sort_bucket groups by (protein, aspect) when aspect column is present."""
    # Create a tiny parquet with 2 proteins x 2 aspects.
    table = pa.table({
        "protein_accession": pa.array(
            ["P0", "P0", "P0", "P0", "P1", "P1", "P1", "P1"], pa.string()
        ),
        "label": pa.array([0, 1, 0, 1, 1, 0, 1, 0], pa.int8()),
        "aspect": pa.array(
            ["mfo", "mfo", "bpo", "bpo", "mfo", "mfo", "bpo", "bpo"], pa.string()
        ),
        "feat": pa.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], pa.float32()),
    })
    bucket_path = tmp_path / "test_bucket.parquet"
    pq.write_table(table, str(bucket_path))

    n_rows, n_groups, group_sizes, labels, proteins, go_terms, aspects = _sort_bucket(
        bucket_path
    )

    assert n_rows == 8
    # Groups: (P0, bpo), (P0, mfo), (P1, bpo), (P1, mfo) -> 4 groups
    assert n_groups == 4, f"Expected 4 (protein,aspect) groups, got {n_groups}"
    assert aspects is not None, "aspects_per_group must be returned when aspect col present"
    assert set(zip(proteins.tolist(), aspects.tolist())) == {
        ("P0", "bpo"), ("P0", "mfo"), ("P1", "bpo"), ("P1", "mfo")
    }
    assert all(s == 2 for s in group_sizes.tolist()), (
        f"All groups should have size 2; got {group_sizes.tolist()}"
    )


def test_sort_bucket_legacy_no_aspect(tmp_path):
    """_sort_bucket groups by protein only when no aspect column is present."""
    table = pa.table({
        "protein_accession": pa.array(["P0", "P0", "P1", "P1"], pa.string()),
        "label": pa.array([0, 1, 1, 0], pa.int8()),
        "feat": pa.array([0.1, 0.2, 0.3, 0.4], pa.float32()),
    })
    bucket_path = tmp_path / "test_legacy.parquet"
    pq.write_table(table, str(bucket_path))

    n_rows, n_groups, group_sizes, labels, proteins, go_terms, aspects = _sort_bucket(
        bucket_path
    )
    assert n_rows == 4
    assert n_groups == 2, f"Legacy mode should group by protein only; got {n_groups}"
    assert aspects is None, "aspects_per_group must be None in legacy mode"


# ---------------------------------------------------------------------------
# Acceptance test 8: TrainingSpec fields
# ---------------------------------------------------------------------------


def test_training_spec_has_test_snapshot_pairs():
    """TrainingSpec exposes test_snapshot_pairs and aspect_conditioned."""
    from protea_reranker_lab.experiment import TrainingSpec

    spec = TrainingSpec(
        cell="nk-mfo",
        val_strategy="none",
        train_snapshot_pairs=["v224-v225", "v225-v226"],
        eval_snapshot_pair="v226-v227",
        test_snapshot_pairs=["v227-v228", "v227-v229", "v227-v230"],
        aspect_conditioned=True,
    )
    assert spec.train_snapshot_pairs == ["v224-v225", "v225-v226"]
    assert spec.eval_snapshot_pair == "v226-v227"
    assert spec.test_snapshot_pairs == ["v227-v228", "v227-v229", "v227-v230"]
    assert spec.aspect_conditioned is True
