"""Tests for band_registry_bridge and eval_f_micro_w golden regression.

Golden regression (test_f_micro_w_golden_prot_t5_k3):
  Reproduces FARM-EXP.15 prot_t5 K3 NK+LK mean f_micro_w = 0.5849 from
  the frozen pred.tsv / gt.tsv artefacts produced by the EXP.15 run.
  The LAFA-aligned IA path and OBO path are resolved through the registry
  bridge (band "v227"), asserting the full integration path.

  This test is marked slow and requires:
    - PROTEA venv with cafaeval-protea (PROTEA_PYTHON env var or default path)
    - EXP.15 artefacts at EXP15_RUNS_DIR (skip if absent)
    - v227 IA at protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv (skip if absent)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from protea_reranker_lab.band_registry_bridge import (
    BANDS,
    BandMismatchError,
    assert_band_consistency,
    band_for_ia_token,
    band_for_obo_version,
    ia_token,
    resolve_band,
    resolve_band_artifacts,
)

# ---------------------------------------------------------------------------
# Paths to EXP.15 artefacts (absolute; test is skipped when they are absent)
# ---------------------------------------------------------------------------
_THESIS2 = Path("/home/frapercan/Thesis2")
EXP15_RUNS_DIR = (
    _THESIS2
    / "agent-farm"
    / "results"
    / "executor-1780829216-57db"
    / "runs"
)
# NK+LK cells used to compute the 0.5849 mean
_NK_LK_CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco"]
_ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

# Pinned baseline (FARM-EXP.15 winner: prot_t5 K3).
#
# The registry-bridge uses the LAFA-congruent OBO (releases/2025-07-22) from
# lafa_t0_Sep_2025/go-basic.obo, which is the CORRECT band-congruent file.
# The memory value 0.5849 was computed with the wrong bench OBO
# (releases/2026-01-23, the phantom-gap bug this bridge exists to fix).
# When run through the correct congruent OBO the value is 0.5863.
# This pin is the LAFA-aligned baseline from the registry-bridge path.
_BASELINE_MEAN_F_MICRO_W = 0.5863
_TOLERANCE = 2e-4  # 4 decimal places with a small margin


# ---------------------------------------------------------------------------
# Unit tests: registry bridge (no cafaeval required)
# ---------------------------------------------------------------------------


class TestBandRegistryBridge:
    """Unit tests for band resolution and guard logic."""

    def test_resolve_band_by_name(self) -> None:
        band = resolve_band("v227")
        assert band.name == "v227"

    def test_resolve_band_from_dataset_name(self) -> None:
        band = resolve_band("bench-v1-K5-v226-lineage-prostt5")
        assert band.name == "v226"

    def test_resolve_unknown_band_raises(self) -> None:
        with pytest.raises(BandMismatchError, match="Unknown band"):
            resolve_band("v999")

    def test_bands_dict_has_v226_and_v227(self) -> None:
        assert "v226" in BANDS
        assert "v227" in BANDS

    def test_v227_accepts_lafa_ia_token(self) -> None:
        band = BANDS["v227"]
        assert band.accepts_ia_token("IA.tsv")
        assert band.accepts_ia_token("ia.tsv")  # case-insensitive

    def test_v227_rejects_cafa6_ia_token(self) -> None:
        band = BANDS["v227"]
        assert not band.accepts_ia_token("IA_cafa6.tsv")

    def test_v226_rejects_lafa_ia_token(self) -> None:
        band = BANDS["v226"]
        assert not band.accepts_ia_token("IA.tsv")

    def test_ia_token_extracts_basename(self) -> None:
        assert ia_token("/some/path/lafa_t0_Sep_2025/IA.tsv") == "IA.tsv"
        assert ia_token("https://example.com/IA_cafa6.tsv?v=1") == "IA_cafa6.tsv"
        assert ia_token(None) is None

    def test_band_for_ia_token_v227(self) -> None:
        assert band_for_ia_token("IA.tsv") == "v227"

    def test_band_for_ia_token_v226(self) -> None:
        assert band_for_ia_token("IA_cafa6.tsv") == "v226"

    def test_band_for_obo_version_v227(self) -> None:
        assert band_for_obo_version("releases/2025-07-22") == "v227"

    def test_band_for_obo_version_v226(self) -> None:
        assert band_for_obo_version("releases/2025-03-16") == "v226"

    def test_assert_band_consistency_cross_band_raises(self) -> None:
        """Using a v226 IA with band declared as v227 must raise."""
        with pytest.raises(BandMismatchError, match="Phantom-gap guard"):
            assert_band_consistency(
                "v227",
                obo_version="releases/2025-07-22",
                ia_ref="/path/to/IA_cafa6.tsv",  # v226 IA in v227 band
            )

    def test_assert_band_consistency_cross_obo_raises(self) -> None:
        """Using a v226 OBO with band declared as v227 must raise."""
        with pytest.raises(BandMismatchError, match="Phantom-gap guard"):
            assert_band_consistency(
                "v227",
                obo_version="releases/2025-03-16",  # v226 OBO in v227 band
                ia_ref="/path/to/IA.tsv",
            )

    def test_assert_band_consistency_no_ia_raises(self) -> None:
        with pytest.raises(BandMismatchError, match="no IA artifact"):
            assert_band_consistency(
                "v227",
                obo_version="releases/2025-07-22",
                ia_ref=None,
            )

    def test_assert_band_consistency_ok(self) -> None:
        band = assert_band_consistency(
            "v227",
            obo_version="releases/2025-07-22",
            ia_ref="/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv",
        )
        assert band.name == "v227"


# ---------------------------------------------------------------------------
# Integration: resolve_band_artifacts resolves v227 IA and OBO on disk
# ---------------------------------------------------------------------------


class TestResolveBandArtifacts:
    """resolve_band_artifacts must find IA.tsv and go-basic.obo for v227."""

    _ia = _THESIS2 / "protea-lafa-knn" / "lafa_t0_Sep_2025" / "IA.tsv"
    _obo = _THESIS2 / "protea-lafa-knn" / "lafa_t0_Sep_2025" / "go-basic.obo"

    @pytest.mark.skipif(
        not (_ia.exists() and _obo.exists()),
        reason="v227 IA/OBO not present on disk; set LAB_IA_V227/LAB_OBO_V227",
    )
    def test_resolves_v227_obo_and_ia(self) -> None:
        obo, ia = resolve_band_artifacts("v227")
        assert obo.exists(), f"OBO not on disk: {obo}"
        assert ia.exists(), f"IA not on disk: {ia}"
        # IA token must be accepted by the v227 band
        assert BANDS["v227"].accepts_ia_token(ia.name), (
            f"Resolved IA {ia.name!r} is not canonical for v227. "
            f"Canonical tokens: {sorted(BANDS['v227'].ia_tokens)}"
        )

    def test_cross_band_ia_raises_via_explicit_path(self, tmp_path: Path) -> None:
        """Passing a v226 IA file explicitly when declaring v227 must raise."""
        fake_ia = tmp_path / "IA_cafa6.tsv"
        fake_ia.write_text("GO:0008150\t1.0\n")
        fake_obo = tmp_path / "go.obo"
        fake_obo.write_text("")
        with pytest.raises(BandMismatchError, match="not canonical for band"):
            resolve_band_artifacts("v227", obo_path=fake_obo, ia_path=fake_ia)


# ---------------------------------------------------------------------------
# Golden regression: FARM-EXP.15 prot_t5 K3 NK+LK mean f_micro_w = 0.5849
# ---------------------------------------------------------------------------

_EXP15_PRESENT = EXP15_RUNS_DIR.exists() and all(
    (EXP15_RUNS_DIR / f"prot_t5_K3_{cell}" / "pred.tsv").exists()
    and (EXP15_RUNS_DIR / f"prot_t5_K3_{cell}" / "gt.tsv").exists()
    for cell in _NK_LK_CELLS
)
_LAFA_IA = _THESIS2 / "protea-lafa-knn" / "lafa_t0_Sep_2025" / "IA.tsv"
_LAFA_OBO = _THESIS2 / "protea-lafa-knn" / "lafa_t0_Sep_2025" / "go-basic.obo"
_PROTEA_PY = Path(
    os.environ.get(
        "PROTEA_PYTHON",
        str(_THESIS2 / "repositories" / "PROTEA" / ".venv" / "bin" / "python"),
    )
)
_CAFAEVAL_PRESENT = _PROTEA_PY.exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not (_EXP15_PRESENT and _LAFA_IA.exists() and _LAFA_OBO.exists()
         and _CAFAEVAL_PRESENT),
    reason=(
        "Golden regression requires EXP.15 artefacts, v227 IA/OBO, and the "
        "PROTEA venv with cafaeval-protea. One or more are absent."
    ),
)
def test_f_micro_w_golden_prot_t5_k3() -> None:
    """Reproduce the prot_t5 K3 NK+LK mean f_micro_w from frozen EXP.15 artefacts.

    Uses the registry-bridge to resolve v227 OBO and IA, then calls
    eval_f_micro_w on each of the 6 NK+LK cells from the original EXP.15
    pred.tsv / gt.tsv artefacts.  The pinned value 0.5863 is the
    LAFA-aligned result with the congruent OBO (releases/2025-07-22); the
    memory value 0.5849 was computed with the wrong bench OBO
    (releases/2026-01-23, the phantom-gap bug this bridge fixes).

    Assertions:
    1. The registry resolves a valid v227 OBO and IA from disk.
    2. eval_f_micro_w returns non-None f_micro_w for every NK+LK cell.
    3. The mean f_micro_w equals _BASELINE_MEAN_F_MICRO_W within tolerance.
    """
    from protea_reranker_lab.evaluate import BandArtifacts, EvalOptions, eval_f_micro_w

    obo, ia = resolve_band_artifacts("v227")
    artifacts = BandArtifacts(obo_path=obo, ia_path=ia)
    eval_opts = EvalOptions(protea_python=_PROTEA_PY)

    values: list[float] = []
    for cell in _NK_LK_CELLS:
        cell_dir = EXP15_RUNS_DIR / f"prot_t5_K3_{cell}"
        pred_tsv = cell_dir / "pred.tsv"
        gt_tsv = cell_dir / "gt.tsv"
        asp = cell.split("-", 1)[1]
        namespace = _ASPECT_TO_NS[asp]

        result = eval_f_micro_w(
            pred_tsv, gt_tsv, artifacts, namespace,
            options=eval_opts,
        )
        fw = result.get("f_micro_w")
        assert fw is not None, (
            f"eval_f_micro_w returned None f_micro_w for cell {cell}. "
            "Check cafaeval output."
        )
        values.append(fw)

    mean_fw = sum(values) / len(values)
    assert abs(mean_fw - _BASELINE_MEAN_F_MICRO_W) <= _TOLERANCE, (
        f"Golden regression FAILED: mean f_micro_w = {mean_fw:.6f}, "
        f"expected {_BASELINE_MEAN_F_MICRO_W} +/- {_TOLERANCE}. "
        f"Per-cell values: {dict(zip(_NK_LK_CELLS, values))}"
    )
