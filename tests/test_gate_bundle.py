"""Reading a frozen pool so the gates can run where the card is.

The bundle exists because the two encoder gates need a reference pool, the pool
lives in a database on the other machine, and this machine is never pointed at it.
So the risk is not that the loader crashes. The risk is that it returns something
subtly different from what the database path returns, and the gate then decides
the fate of the whole programme on a pool nobody can reconstruct.

These tests pin the three things that would be silent: that the caller cannot tell
the two paths apart, that a bundle without evidence of the sequence-twin exclusion
is refused rather than used, and that a row-count mismatch stops rather than
attributing embeddings to the wrong proteins.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from protea_reranker_lab.gate_bundle import (
    bundle_arm_data,
    describe,
    load_gate_bundle,
)


def _propagate(leaves, _dag):
    """Stands in for the ontology closure: a term set, empty for the unannotated."""
    return frozenset(leaves)


def _write(tmp_path, *, refs=2, queries=1, dim=3, leaves=None, manifest=None, ragged=0):
    path = tmp_path / "bundle.npz"
    ref_accs = [f"R{i}" for i in range(refs)]
    q_accs = [f"Q{i}" for i in range(queries)]
    full = {
        "embedding_config_id": "cfg",
        "annotation_set_id": "ann",
        "excluded_by_sequence": 7,
        "ref_n_written": refs,
        "queries_written": queries,
        "dim": dim,
        "seed": 42,
    }
    if manifest is not None:
        full = manifest
    np.savez_compressed(
        path,
        reference_accessions=np.array(ref_accs, dtype=object),
        reference_embeddings=np.ones((refs + ragged, dim), dtype=np.float32),
        query_accessions=np.array(q_accs, dtype=object),
        query_embeddings=np.full((queries, dim), 2.0, dtype=np.float32),
        reference_leaves=json.dumps(
            leaves if leaves is not None else {a: ["GO:1"] for a in ref_accs}
        ),
        manifest=json.dumps(full),
    )
    return path


# --------------------------------------------------------------------------- loading

def test_a_well_formed_bundle_loads(tmp_path):
    bundle = load_gate_bundle(_write(tmp_path))

    assert len(bundle.reference_accessions) == 2
    assert bundle.dim == 3
    assert bundle.manifest["excluded_by_sequence"] == 7


def test_embeddings_come_back_as_float32(tmp_path):
    """The ablation vstacks these; a float64 pool doubles memory silently."""
    bundle = load_gate_bundle(_write(tmp_path))

    assert bundle.reference_embeddings.dtype == np.float32
    assert bundle.query_embeddings.dtype == np.float32


# ------------------------------------------------------------------ the refusals

def test_a_bundle_without_the_twin_exclusion_is_refused(tmp_path):
    """The load-bearing refusal.

    A producer predating the sequence-level exclusion emits a pool in which every
    query can retrieve its own vector at cosine 1.0. That is a guaranteed rank
    one, and at K=3 it is a third of the vote handed straight to the answer. It
    would not fail anywhere; it would just win.
    """
    manifest = {"embedding_config_id": "c", "annotation_set_id": "a",
                "ref_n_written": 2, "dim": 3}

    with pytest.raises(ValueError, match="excluded_by_sequence"):
        load_gate_bundle(_write(tmp_path, manifest=manifest))


def test_the_refusal_says_why_it_matters(tmp_path):
    manifest = {"embedding_config_id": "c", "annotation_set_id": "a",
                "ref_n_written": 2, "dim": 3}

    with pytest.raises(ValueError) as excinfo:
        load_gate_bundle(_write(tmp_path, manifest=manifest))

    assert "cosine 1.0" in str(excinfo.value)


def test_more_rows_than_accessions_stops_rather_than_misattributing(tmp_path):
    """Numpy will index a mismatched pair happily and every row lands on the
    wrong protein, which is the shape of the join corruption that once rewrote
    8.3 per cent of rows."""
    with pytest.raises(ValueError, match="reference accessions"):
        load_gate_bundle(_write(tmp_path, ragged=1))


def test_a_width_mismatch_between_queries_and_references_stops(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        reference_accessions=np.array(["R0"], dtype=object),
        reference_embeddings=np.ones((1, 4), dtype=np.float32),
        query_accessions=np.array(["Q0"], dtype=object),
        query_embeddings=np.ones((1, 3), dtype=np.float32),
        reference_leaves=json.dumps({"R0": ["GO:1"]}),
        manifest=json.dumps({"embedding_config_id": "c", "annotation_set_id": "a",
                             "excluded_by_sequence": 0, "ref_n_written": 1, "dim": 4}),
    )

    with pytest.raises(ValueError, match="dimensional"):
        load_gate_bundle(path)


# ------------------------------------------------------------------- the contract

def test_it_returns_the_same_four_things_the_database_path_returns(tmp_path):
    bundle = load_gate_bundle(_write(tmp_path, refs=3, queries=2, dim=4))

    R, Q, ref_clo, q_accs = bundle_arm_data(bundle, object(), _propagate)

    assert R.shape == (3, 4)
    assert Q.shape == (2, 4)
    assert len(ref_clo) == 3
    assert q_accs == ["Q0", "Q1"]


def test_a_reference_without_a_closure_is_dropped_as_the_database_path_drops_it(tmp_path):
    """Same rule on both sides, or the two pools differ by their unannotated tail."""
    leaves = {"R0": ["GO:1"], "R1": [], "R2": ["GO:2"]}
    bundle = load_gate_bundle(_write(tmp_path, refs=3, leaves=leaves))

    R, _Q, ref_clo, _q = bundle_arm_data(bundle, object(), _propagate)

    assert R.shape[0] == 2
    assert len(ref_clo) == 2


def test_rows_stay_aligned_with_their_closures_after_dropping(tmp_path):
    """The drop rebuilds both lists from the same filtered accessions, so a
    reference's vector and its closure cannot come apart."""
    leaves = {"R0": [], "R1": ["GO:9"]}
    bundle = load_gate_bundle(_write(tmp_path, refs=2, dim=2, leaves=leaves))
    bundle.reference_embeddings[1] = np.array([5.0, 5.0], dtype=np.float32)

    R, _Q, ref_clo, _q = bundle_arm_data(bundle, object(), _propagate)

    assert R.shape[0] == 1
    assert np.allclose(R[0], [5.0, 5.0])
    assert ref_clo[0] == frozenset({"GO:9"})


def test_a_pool_with_no_closures_at_all_stops(tmp_path):
    """Almost always the annotation set and the ontology are not the recorded pair."""
    leaves = {"R0": [], "R1": []}
    bundle = load_gate_bundle(_write(tmp_path, refs=2, leaves=leaves))

    with pytest.raises(ValueError, match="propagable closure"):
        bundle_arm_data(bundle, object(), _propagate)


def test_the_pool_is_not_resampled(tmp_path):
    """The bundle is already an ordered draw plus a seeded sample. A second draw
    here would mean the manifest describes a pool that is not the one in use."""
    bundle = load_gate_bundle(_write(tmp_path, refs=4))

    R_a, *_ = bundle_arm_data(bundle, object(), _propagate)
    R_b, *_ = bundle_arm_data(bundle, object(), _propagate)

    assert R_a.shape[0] == 4
    assert np.array_equal(R_a, R_b)


# ------------------------------------------------------------------------ the log

def test_describe_names_the_pool_that_produced_a_result(tmp_path):
    bundle = load_gate_bundle(_write(tmp_path))

    line = describe(bundle)

    assert "cfg" in line and "twins_excluded=7" in line
