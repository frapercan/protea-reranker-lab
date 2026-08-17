"""The lab used to complete without reporting. These are the tests that fail if it can again.

Two defects are covered. Host locations were absolute paths into one developer's
home directory, so after the machine was reinstalled under a different user name
they resolved to nothing. And a cell the scorer could not produce was logged as
NA, omitted from the results, and the run still wrote ``status: ok``. Together
those meant a run could look successful with every metric empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from protea_reranker_lab import host_paths
from protea_reranker_lab.encoder_ablation import (
    CellNotScored,
    EncoderAblationSpec,
    _eval_arm_cells,
)


# --------------------------------------------------------------------------- host paths

def test_environment_variable_wins_over_every_fallback(tmp_path, monkeypatch):
    wanted = tmp_path / "releases" / "Sep_2025_Mar_2026"
    wanted.mkdir(parents=True)
    monkeypatch.setenv("PROTEA_LAB_GT_DIR", str(wanted))
    assert host_paths.ground_truth_dir() == wanted


def test_missing_location_raises_and_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("PROTEA_LAB_GT_DIR", raising=False)
    monkeypatch.setenv("THESIS_ROOT", str(tmp_path))  # empty root, nothing to find

    with pytest.raises(host_paths.MissingHostPath) as excinfo:
        host_paths.ground_truth_dir()

    assert excinfo.value.env_var == "PROTEA_LAB_GT_DIR"
    assert "PROTEA_LAB_GT_DIR" in str(excinfo.value)
    # The message has to carry what was tried, or the technician cannot tell a
    # wrong variable from a missing directory.
    assert str(tmp_path) in str(excinfo.value)


def test_thesis_root_is_derived_from_the_checkout_not_from_a_home_directory(monkeypatch):
    monkeypatch.delenv("THESIS_ROOT", raising=False)
    root = host_paths.thesis_root()
    assert root.is_absolute()
    # The failure this guards against is a path under someone's home being
    # baked in again. Deriving from __file__ keeps a moved checkout working.
    assert "frapercan" not in str(root)


def test_interpreter_falls_back_to_the_running_one(tmp_path, monkeypatch):
    import sys

    monkeypatch.delenv("PROTEA_LAB_PYTHON", raising=False)
    monkeypatch.setenv("THESIS_ROOT", str(tmp_path))
    assert host_paths.protea_python() == Path(sys.executable)


# --------------------------------------------------------------------------- the spec

def test_spec_leaves_host_paths_unset_until_asked():
    spec = EncoderAblationSpec()
    assert spec.gt_dir is None
    assert spec.protea_python is None


def test_resolve_fills_both_and_does_so_before_any_work(tmp_path, monkeypatch):
    gt = tmp_path / "gt"
    gt.mkdir()
    monkeypatch.setenv("PROTEA_LAB_GT_DIR", str(gt))
    monkeypatch.setenv("THESIS_ROOT", str(tmp_path))

    spec = EncoderAblationSpec()
    spec.resolve_host_paths()

    assert spec.gt_dir == gt
    assert spec.protea_python is not None


def test_resolve_is_idempotent_and_respects_an_explicit_value(tmp_path, monkeypatch):
    explicit = tmp_path / "chosen"
    explicit.mkdir()
    monkeypatch.setenv("PROTEA_LAB_GT_DIR", str(tmp_path / "ignored"))
    monkeypatch.setenv("THESIS_ROOT", str(tmp_path))

    spec = EncoderAblationSpec(gt_dir=explicit)
    spec.resolve_host_paths()
    spec.resolve_host_paths()

    assert spec.gt_dir == explicit


# --------------------------------------------------------------------------- the silent cell

def _spec_for_scoring(tmp_path: Path) -> EncoderAblationSpec:
    return EncoderAblationSpec(
        gt_dir=tmp_path, protea_python=Path("/nonexistent/python"),
        official_harness=False,
    )


def test_a_cell_the_scorer_could_not_produce_stops_the_run(tmp_path, monkeypatch):
    """The regression. This exact shape completed with status ok."""
    monkeypatch.setattr(
        "protea_reranker_lab.encoder_ablation._run_cafaeval",
        lambda *a, **k: {"error": "cafaeval exited 1: No such file or directory"},
    )
    spec = _spec_for_scoring(tmp_path)

    with pytest.raises(CellNotScored) as excinfo:
        _eval_arm_cells(tmp_path, [("nk", "bpo")], tmp_path / "go.obo", tmp_path / "ia.tsv",
                        spec, {})

    message = str(excinfo.value)
    assert "nk-bpo" in message
    # The scorer's own words must survive to the exception, since the previous
    # code truncated a CalledProcessError whose text never named the cause.
    assert "No such file or directory" in message


def test_output_without_the_requested_aspect_also_stops_the_run(tmp_path, monkeypatch):
    """A scorer that ran but wrote no row for this aspect is not a zero score."""
    monkeypatch.setattr(
        "protea_reranker_lab.encoder_ablation._run_cafaeval",
        lambda *a, **k: {"f_micro_w": None},
    )
    spec = _spec_for_scoring(tmp_path)

    with pytest.raises(CellNotScored):
        _eval_arm_cells(tmp_path, [("lk", "mfo")], tmp_path / "go.obo", tmp_path / "ia.tsv",
                        spec, {})


def test_scored_cells_are_returned_and_a_later_failure_still_stops(tmp_path, monkeypatch):
    """A run must not report the cells that worked and drop the ones that did not."""
    seen: list[str] = []

    def fake(cell, *a, **k):
        seen.append(cell)
        return {"f_micro_w": 0.31} if cell == "nk-bpo" else {"error": "boom"}

    monkeypatch.setattr("protea_reranker_lab.encoder_ablation._run_cafaeval", fake)
    spec = _spec_for_scoring(tmp_path)

    with pytest.raises(CellNotScored):
        _eval_arm_cells(tmp_path, [("nk", "bpo"), ("nk", "cco")], tmp_path / "go.obo",
                        tmp_path / "ia.tsv", spec, {})

    assert seen == ["nk-bpo", "nk-cco"]


def test_all_cells_scored_returns_every_value(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "protea_reranker_lab.encoder_ablation._run_cafaeval",
        lambda cell, *a, **k: {"f_micro_w": 0.25 if cell.startswith("nk") else 0.40},
    )
    spec = _spec_for_scoring(tmp_path)

    got = _eval_arm_cells(tmp_path, [("nk", "bpo"), ("pk", "mfo")], tmp_path / "go.obo",
                          tmp_path / "ia.tsv", spec, {})

    assert got == {"nk-bpo": 0.25, "pk-mfo": 0.40}


# --------------------------------------------------------------------------- run identity

def test_harness_mode_changes_the_run_identity():
    """Two experiments must not share an output directory.

    ``spec_hash`` keys the output directory. With the harness fields absent from
    the payload, a plain run and an official-harness run of the same settings
    hashed identically, wrote to the same place, and the second overwrote the
    first without a word.
    """
    plain = EncoderAblationSpec(official_harness=False)
    official = EncoderAblationSpec(official_harness=True)
    assert plain.spec_hash() != official.spec_hash()


def test_the_terms_of_interest_file_changes_the_run_identity(tmp_path):
    a = EncoderAblationSpec(official_harness=True, toi_path=tmp_path / "toi_a.tsv")
    b = EncoderAblationSpec(official_harness=True, toi_path=tmp_path / "toi_b.tsv")
    assert a.spec_hash() != b.spec_hash()


def test_host_paths_do_not_change_the_run_identity(tmp_path):
    """Where the ground truth lives is a property of the machine, not the experiment."""
    a = EncoderAblationSpec(gt_dir=tmp_path / "one")
    b = EncoderAblationSpec(gt_dir=tmp_path / "two")
    assert a.spec_hash() == b.spec_hash()
