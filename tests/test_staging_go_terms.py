"""``carry_go_terms`` emits a row-aligned ``go_terms.npy`` per split.

Palanca-1 IA weighting maps each training row to ``IA(go)``; that requires
staging to persist the GO term per row in the *same* order as ``labels.npy``
(rows bucketed by protein, then stable-sorted within a bucket). This test
proves the alignment end-to-end on a tiny synthetic dump and proves the
default path (``carry_go_terms=False``) emits no ``go_terms.npy``.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from protea_reranker_lab.staging import stage_for_training


def _write_synthetic(path, *, n_proteins=12, k=4, seed=0):
    rng = np.random.default_rng(seed)
    rows_prot, rows_label, rows_go, rows_feat = [], [], [], []
    for pidx in range(n_proteins):
        acc = f"P{pidx:05d}"
        for j in range(k):
            rows_prot.append(acc)
            rows_label.append(int(rng.random() < 0.4))
            rows_go.append(f"GO:{pidx:04d}{j:03d}")
            rows_feat.append(float(rng.random()))
    table = pa.table({
        "protein_accession": pa.array(rows_prot, pa.string()),
        "label": pa.array(rows_label, pa.int8()),
        "go_term_id": pa.array(rows_go, pa.string()),
        "category": pa.array(["lk"] * len(rows_prot), pa.string()),
        "aspect": pa.array(["bpo"] * len(rows_prot), pa.string()),
        "snapshot_pair": pa.array(["v226-v230"] * len(rows_prot), pa.string()),
        "knn_vote_score": pa.array(rows_feat, pa.float32()),
    })
    pq.write_table(table, str(path))
    # the (acc, go) -> label truth map, used to verify alignment
    truth = {(p, g): lab for p, g, lab in zip(rows_prot, rows_go, rows_label)}
    return truth


@pytest.fixture
def dataset(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    truth = _write_synthetic(ds / "train.parquet", seed=1)
    _write_synthetic(ds / "eval.parquet", seed=2)
    return ds, truth


def _stage(ds_dir, out_dir, *, carry):
    return stage_for_training(
        source_train_parquet=ds_dir / "train.parquet",
        source_eval_parquet=ds_dir / "eval.parquet",
        cell=("lk", "bpo"),
        feature_cols=["knn_vote_score"],
        categorical_cols=[],
        out_dir=out_dir,
        val_strategy="protein_group",
        val_fraction=0.25,
        val_holdout_snapshot=None,
        neg_pos_ratio=None,
        seed=42,
        carry_go_terms=carry,
    )


def test_go_terms_emitted_and_row_aligned(dataset, tmp_path):
    ds_dir, truth = dataset
    stage = _stage(ds_dir, tmp_path / "stage", carry=True)

    assert stage.train.go_terms_path is not None
    go = np.load(stage.train.go_terms_path, allow_pickle=True)
    labels = np.load(stage.train.labels_path)
    groups = np.load(stage.train.groups_path)
    proteins_per_group = np.load(stage.train.proteins_path, allow_pickle=True)
    proteins = np.repeat(proteins_per_group, groups)

    assert go.shape[0] == labels.shape[0] == proteins.shape[0]
    # every (protein, go) row carries the label the source assigned it
    for p, g, lab in zip(proteins, go, labels):
        assert truth[(p, g)] == int(lab)

    # eval split never carries go terms
    assert stage.eval.go_terms_path is None


def test_default_path_emits_no_go_terms(dataset, tmp_path):
    ds_dir, _ = dataset
    stage = _stage(ds_dir, tmp_path / "stage", carry=False)
    assert stage.train.go_terms_path is None
    assert (stage.val is None) or (stage.val.go_terms_path is None)
