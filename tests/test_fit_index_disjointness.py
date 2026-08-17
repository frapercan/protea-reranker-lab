"""The gate that asks whether the learned arm generalises or memorises the bank.

The shipped configuration fits the encoder on the reference pool and then scores
it by retrieving from that same pool: ``closures`` and ``ref_closures`` were the
same object. So the encoder has seen every donor it will be measured against,
and a gain under that condition is not a generalisation result.

The obvious version of this test fits on half and retrieves from the other half,
then compares against the shipped condition which uses the whole pool for both.
That comparison is confounded: the disjoint arm would have half the donors, and a
smaller bank retrieves worse for reasons that have nothing to do with what the
encoder saw. Both arms therefore halve the pool, and differ in one thing only.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.encoder_ablation import (
    ArmData,
    ArmSpec,
    EncoderAblationSpec,
    _build_arm,
    split_fit_index,
)


# --------------------------------------------------------------------------- the split

def test_shared_mode_fits_and_retrieves_on_the_same_rows():
    fit, index = split_fit_index(100, "shared", seed=42)
    assert np.array_equal(fit, index)


def test_disjoint_mode_shares_no_row():
    fit, index = split_fit_index(100, "disjoint", seed=42)
    assert set(fit.tolist()).isdisjoint(index.tolist())


def test_both_modes_use_the_same_sizes_which_is_the_whole_point():
    """The control. Differing sizes would measure index size, not disjointness."""
    shared_fit, shared_index = split_fit_index(100, "shared", seed=42)
    disjoint_fit, disjoint_index = split_fit_index(100, "disjoint", seed=42)

    assert len(shared_fit) == len(disjoint_fit)
    assert len(shared_index) == len(disjoint_index)


def test_the_fit_pool_is_identical_across_modes():
    """Only the index moves, so a difference cannot come from a different fit."""
    shared_fit, _ = split_fit_index(100, "shared", seed=42)
    disjoint_fit, _ = split_fit_index(100, "disjoint", seed=42)

    assert np.array_equal(shared_fit, disjoint_fit)


def test_the_split_is_deterministic_under_a_seed():
    a = split_fit_index(1000, "disjoint", seed=7)
    b = split_fit_index(1000, "disjoint", seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_a_different_seed_gives_a_different_split():
    a, _ = split_fit_index(1000, "disjoint", seed=7)
    b, _ = split_fit_index(1000, "disjoint", seed=8)
    assert not np.array_equal(a, b)


def test_an_odd_pool_still_leaves_the_halves_disjoint():
    fit, index = split_fit_index(101, "disjoint", seed=1)
    assert set(fit.tolist()).isdisjoint(index.tolist())
    assert len(fit) + len(index) == 101


def test_an_unknown_mode_is_refused_rather_than_defaulted():
    """Silently falling back to shared would report a leak as a clean result."""
    with pytest.raises(ValueError, match="unknown fit_index_mode"):
        split_fit_index(100, "held-out", seed=42)


def test_the_mode_changes_the_run_identity():
    """The two conditions are different experiments and must not share a directory."""
    shared = EncoderAblationSpec(fit_index_mode="shared")
    disjoint = EncoderAblationSpec(fit_index_mode="disjoint")
    assert shared.spec_hash() != disjoint.spec_hash()


def test_the_default_reproduces_the_shipped_condition():
    """So that adopting this change alone moves no existing number."""
    assert EncoderAblationSpec().fit_index_mode == "shared"


# --------------------------------------------------------------------------- fit versus apply

def test_the_arm_is_fit_on_the_subset_and_applied_to_everything(monkeypatch):
    """Separating fit from apply is the mechanism; this asserts it is wired that way."""
    seen: dict[str, int] = {}

    def fake_fit(R, closures, dag, arm, spec):  # noqa: ARG001
        seen["fit_rows"] = R.shape[0]
        seen["fit_closures"] = len(closures)
        return object()

    def fake_apply(enc, X, top_k):  # noqa: ARG001
        return X

    monkeypatch.setattr("protea_reranker_lab.encoder_ablation.fit_encoder", fake_fit)
    monkeypatch.setattr("protea_reranker_lab.encoder_ablation.apply_encoder", fake_apply)

    R = np.arange(40, dtype=np.float32).reshape(20, 2)
    Q = np.zeros((3, 2), dtype=np.float32)
    closures = [frozenset({f"GO:{i}"}) for i in range(20)]
    fit_rows = np.arange(10)

    Rx, _ = _build_arm(
        ArmSpec(name="learned", kind="learned"), ArmData(R, Q, closures, None),
        EncoderAblationSpec(), fit_rows,
    )

    assert seen["fit_rows"] == 10
    assert seen["fit_closures"] == 10
    # applied to every reference row, not only the fitted ones, because
    # retrieval draws from rows the encoder was not fit on
    assert Rx.shape[0] == 20


def test_without_a_mask_the_arm_fits_on_everything(monkeypatch):
    """The pre-existing behaviour, kept so this change is opt-in."""
    seen: dict[str, int] = {}

    monkeypatch.setattr(
        "protea_reranker_lab.encoder_ablation.fit_encoder",
        lambda R, c, d, a, s: seen.update(fit_rows=R.shape[0]) or object(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "protea_reranker_lab.encoder_ablation.apply_encoder",
        lambda enc, X, k: X,  # noqa: ARG005
    )

    R = np.arange(40, dtype=np.float32).reshape(20, 2)
    closures = [frozenset({f"GO:{i}"}) for i in range(20)]

    _build_arm(
        ArmSpec(name="learned", kind="learned"),
        ArmData(R, np.zeros((2, 2), dtype=np.float32), closures, None),
        EncoderAblationSpec(),
    )

    assert seen["fit_rows"] == 20
