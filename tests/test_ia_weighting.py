"""Unit tests for palanca-1 Information-Accretion sample weighting."""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.ia_weighting import (
    IA_MODES,
    ia_weights,
    load_ia_table,
)


@pytest.fixture
def ia_table(tmp_path):
    p = tmp_path / "ia.txt"
    p.write_text(
        "GO:0000001\t0.0\n"
        "GO:0000011\t0.07038932789139803\n"
        "GO:0000012\t5.992863818718371\n"
        "GO:0000099\t-3.0\n"        # negative -> clamped to 0
        "GO:0000098\tnot_a_float\n"  # unparseable -> skipped
        "badline\n"                  # too few columns -> skipped
    )
    return load_ia_table(p)


def test_load_ia_table_parses_and_clamps(ia_table):
    assert ia_table["GO:0000012"] == pytest.approx(5.992863818718371)
    assert ia_table["GO:0000099"] == 0.0          # negative clamped
    assert "GO:0000098" not in ia_table           # unparseable skipped
    assert "badline" not in ia_table


def test_mode_none_returns_none(ia_table):
    go = np.array(["GO:0000012"], dtype=object)
    lab = np.array([1], dtype=np.int8)
    assert ia_weights(go, lab, ia_table, mode="none") is None


def test_mode_positives_only_lifts_positive_rows(ia_table):
    go = np.array(["GO:0000012", "GO:0000012", "GO:0000011"], dtype=object)
    lab = np.array([1, 0, 1], dtype=np.int8)
    w = ia_weights(go, lab, ia_table, mode="positives")
    # positive high-IA term lifted; negative (even high-IA) stays 1.0
    assert w[0] == pytest.approx(1.0 + 5.992863818718371)
    assert w[1] == pytest.approx(1.0)
    # positive low-IA term lifted by its (small) IA
    assert w[2] == pytest.approx(1.0 + 0.07038932789139803)


def test_mode_all_lifts_every_row_by_its_term_ia(ia_table):
    go = np.array(["GO:0000012", "GO:0000011"], dtype=object)
    lab = np.array([1, 0], dtype=np.int8)
    w = ia_weights(go, lab, ia_table, mode="all")
    assert w[0] == pytest.approx(1.0 + 5.992863818718371)
    assert w[1] == pytest.approx(1.0 + 0.07038932789139803)  # negative weighted too


def test_unknown_term_gets_neutral_weight(ia_table):
    go = np.array(["GO:9999999"], dtype=object)
    lab = np.array([1], dtype=np.int8)
    assert ia_weights(go, lab, ia_table, mode="positives")[0] == pytest.approx(1.0)


def test_scale_is_linear(ia_table):
    go = np.array(["GO:0000012"], dtype=object)
    lab = np.array([1], dtype=np.int8)
    w = ia_weights(go, lab, ia_table, mode="all", scale=2.0)
    assert w[0] == pytest.approx(1.0 + 2.0 * 5.992863818718371)


def test_length_mismatch_raises(ia_table):
    go = np.array(["GO:0000012", "GO:0000011"], dtype=object)
    lab = np.array([1], dtype=np.int8)
    with pytest.raises(ValueError):
        ia_weights(go, lab, ia_table, mode="all")


def test_bad_mode_raises(ia_table):
    go = np.array(["GO:0000012"], dtype=object)
    lab = np.array([1], dtype=np.int8)
    with pytest.raises(ValueError):
        ia_weights(go, lab, ia_table, mode="bogus")


def test_weights_are_finite_and_non_negative(ia_table):
    go = np.array(["GO:0000012", "GO:9999999", "GO:0000099"], dtype=object)
    lab = np.array([1, 0, 1], dtype=np.int8)
    for mode in (m for m in IA_MODES if m != "none"):
        w = ia_weights(go, lab, ia_table, mode=mode)
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)
