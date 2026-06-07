"""Tests for F-RERANK-UNIVERSAL.5d group-key fix.

Acceptance criteria (per slice spec):
1. Within any group, go_term_id is UNIQUE (no intra-group term duplication
   after K-collapse).
2. group count == distinct (snapshot_pair, protein, aspect, plm_id) tuples.
3. A term that is neg in one snapshot and pos in another lands in DIFFERENT
   groups (incoherent-label isolation).
4. The existing per-(protein, aspect) key path still works when snapshot_pair
   is absent from the bucket (backward compat).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from protea_reranker_lab.splits import _sort_bucket, _k_collapse_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bucket_parquet(
    path: Path,
    *,
    proteins: list[str],
    aspects: list[str],
    snapshot_pairs: list[str],
    plm_ids: list[int],
    go_terms: list[str],
    labels: list[int],
    feat: list[float],
) -> None:
    """Write a synthetic bucket parquet for _sort_bucket testing."""
    table = pa.table({
        "protein_accession": pa.array(proteins, pa.string()),
        "label": pa.array(labels, pa.int8()),
        "go_term_id": pa.array(go_terms, pa.string()),
        "aspect": pa.array(aspects, pa.string()),
        "snapshot_pair": pa.array(snapshot_pairs, pa.string()),
        "plm_id": pa.array(plm_ids, pa.int32()),
        "feat": pa.array(feat, pa.float32()),
    })
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Test 1: go_term_id is unique within each group after K-collapse
# ---------------------------------------------------------------------------


def test_intra_group_go_term_unique_after_k_collapse(tmp_path):
    """After _sort_bucket the same go_term must appear at most once per group.

    We simulate K3/K5/K10 by putting the same (protein, snapshot, aspect,
    plm_id, go_term) tuple twice (different feat values), which would happen
    when K5 and K10 both contain the same candidate term.
    """
    # Two rows for the same group+term (simulating K-duplication).
    _make_bucket_parquet(
        tmp_path / "bucket.parquet",
        proteins=["P0", "P0", "P0", "P0"],
        aspects=["mfo", "mfo", "mfo", "mfo"],
        snapshot_pairs=["v226-v227", "v226-v227", "v226-v227", "v226-v227"],
        plm_ids=[0, 0, 0, 0],
        go_terms=["GO:0001", "GO:0001", "GO:0002", "GO:0002"],  # each duplicated
        labels=[1, 1, 0, 0],
        feat=[0.8, 0.7, 0.3, 0.3],
    )
    n_rows, n_groups, group_sizes, labels, proteins, go_terms, aspects = _sort_bucket(
        tmp_path / "bucket.parquet"
    )
    # After K-collapse: 1 group (P0, v226-v227, mfo, plm_id=0), 2 unique terms.
    assert n_groups == 1, f"Expected 1 group, got {n_groups}"
    assert n_rows == 2, f"Expected 2 rows (one per term after dedup), got {n_rows}"
    assert group_sizes[0] == 2


# ---------------------------------------------------------------------------
# Test 2: group count == distinct (snapshot_pair, protein, aspect, plm_id)
# ---------------------------------------------------------------------------


def test_group_count_equals_4tuple_distinct(tmp_path):
    """n_groups must equal distinct (snapshot_pair, protein, aspect, plm_id) tuples."""
    # 2 proteins x 2 snapshots x 2 aspects x 2 plm_ids = 16 groups
    prots = []
    snaps = []
    asps = []
    plms = []
    gos = []
    labs = []
    feats = []
    for p in ["P0", "P1"]:
        for s in ["v225-v226", "v226-v227"]:
            for a in ["mfo", "bpo"]:
                for plm in [0, 1]:
                    for j in range(3):
                        prots.append(p)
                        snaps.append(s)
                        asps.append(a)
                        plms.append(plm)
                        gos.append(f"GO:{p}{s}{a}{plm}{j:02d}")
                        labs.append(j % 2)
                        feats.append(float(j) / 10)

    _make_bucket_parquet(
        tmp_path / "bucket.parquet",
        proteins=prots,
        aspects=asps,
        snapshot_pairs=snaps,
        plm_ids=plms,
        go_terms=gos,
        labels=labs,
        feat=feats,
    )
    n_rows, n_groups, group_sizes, labels, proteins_arr, go_terms, aspects_arr = _sort_bucket(
        tmp_path / "bucket.parquet"
    )

    expected_groups = len({(p, s, a, plm) for p, s, a, plm in zip(prots, snaps, asps, plms)})
    assert n_groups == expected_groups, (
        f"n_groups={n_groups} must equal distinct 4-tuples={expected_groups}"
    )
    # Each group has 3 unique terms.
    assert n_rows == expected_groups * 3, (
        f"Expected {expected_groups * 3} rows, got {n_rows}"
    )


# ---------------------------------------------------------------------------
# Test 3: incoherent-label isolation across snapshots
# ---------------------------------------------------------------------------


def test_neg_in_one_snapshot_pos_in_another_land_in_different_groups(tmp_path):
    """A go_term that is neg in snapshot A and pos in snapshot B must be in
    DIFFERENT groups (one per snapshot)."""
    _make_bucket_parquet(
        tmp_path / "bucket.parquet",
        proteins=["P0", "P0"],
        aspects=["mfo", "mfo"],
        snapshot_pairs=["v225-v226", "v226-v227"],
        plm_ids=[0, 0],
        go_terms=["GO:0001", "GO:0001"],  # same term, different snapshot
        labels=[0, 1],                    # neg in first, pos in second
        feat=[0.3, 0.8],
    )
    n_rows, n_groups, group_sizes, labels, proteins_arr, go_terms, aspects_arr = _sort_bucket(
        tmp_path / "bucket.parquet"
    )
    # Must be 2 groups (one per snapshot), each with 1 term.
    assert n_groups == 2, (
        f"Expected 2 groups (one per snapshot), got {n_groups}"
    )
    assert n_rows == 2, f"Expected 2 rows, got {n_rows}"
    # Labels must differ: one group has label 0, the other has label 1.
    assert set(labels.tolist()) == {0, 1}, (
        f"Labels in the two groups must differ; got {labels.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4: backward compat -- (protein, aspect) key when snapshot_pair absent
# ---------------------------------------------------------------------------


def test_sort_bucket_no_snapshot_uses_protein_aspect_key(tmp_path):
    """When snapshot_pair is absent, group key falls back to (protein, aspect)."""
    table = pa.table({
        "protein_accession": pa.array(["P0", "P0", "P1", "P1"], pa.string()),
        "label": pa.array([0, 1, 1, 0], pa.int8()),
        "aspect": pa.array(["mfo", "mfo", "bpo", "bpo"], pa.string()),
        "feat": pa.array([0.1, 0.2, 0.3, 0.4], pa.float32()),
    })
    bucket_path = tmp_path / "bucket_legacy.parquet"
    pq.write_table(table, str(bucket_path))

    n_rows, n_groups, group_sizes, labels, proteins_arr, go_terms, aspects_arr = _sort_bucket(
        bucket_path
    )
    # 2 proteins x 1 aspect each = 2 groups (no snapshot col -> old key)
    assert n_groups == 2, f"Expected 2 groups, got {n_groups}"
    assert go_terms is None, "go_terms must be None when go_term_id col absent"


# ---------------------------------------------------------------------------
# Test 5: _k_collapse_table correctness
# ---------------------------------------------------------------------------


def test_k_collapse_deduplicates_go_terms_within_group(tmp_path):
    """_k_collapse_table keeps the first occurrence of each go_term per group."""
    table = pa.table({
        "protein_accession": pa.array(["P0", "P0", "P0", "P0"], pa.string()),
        "label": pa.array([1, 0, 0, 1], pa.int8()),
        "go_term_id": pa.array(["GO:0001", "GO:0001", "GO:0002", "GO:0002"], pa.string()),
        "aspect": pa.array(["mfo", "mfo", "mfo", "mfo"], pa.string()),
        "snapshot_pair": pa.array(
            ["v226-v227", "v226-v227", "v226-v227", "v226-v227"], pa.string()
        ),
        "feat": pa.array([0.9, 0.5, 0.3, 0.8], pa.float32()),
    })
    result = _k_collapse_table(table)
    assert result.num_rows == 2, f"Expected 2 rows after dedup, got {result.num_rows}"
    gos = result.column("go_term_id").to_pylist()
    assert sorted(gos) == ["GO:0001", "GO:0002"], f"Wrong go terms after dedup: {gos}"
    # First occurrence kept: GO:0001 label=1, GO:0002 label=0
    label_map = {g: lab for g, lab in zip(gos, result.column("label").to_pylist())}
    assert label_map["GO:0001"] == 1, "First occurrence of GO:0001 has label=1"
    assert label_map["GO:0002"] == 0, "First occurrence of GO:0002 has label=0"


def test_k_collapse_noop_when_no_go_term_id():
    """_k_collapse_table is a no-op when go_term_id column is absent."""
    table = pa.table({
        "protein_accession": pa.array(["P0", "P0"], pa.string()),
        "label": pa.array([0, 1], pa.int8()),
        "feat": pa.array([0.1, 0.2], pa.float32()),
    })
    result = _k_collapse_table(table)
    assert result.num_rows == 2, "No-op: all rows kept when go_term_id absent"


def test_k_collapse_preserves_unique_terms():
    """_k_collapse_table keeps all rows when every term is already unique."""
    table = pa.table({
        "protein_accession": pa.array(["P0", "P0", "P0"], pa.string()),
        "label": pa.array([1, 0, 1], pa.int8()),
        "go_term_id": pa.array(["GO:0001", "GO:0002", "GO:0003"], pa.string()),
        "snapshot_pair": pa.array(["v226-v227"] * 3, pa.string()),
        "feat": pa.array([0.5, 0.3, 0.8], pa.float32()),
    })
    result = _k_collapse_table(table)
    assert result.num_rows == 3, "All rows kept when terms are unique"
