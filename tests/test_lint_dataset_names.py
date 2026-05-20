"""Tests for the FARM-EXP.12 dataset-name linter.

The linter rejects ``bench-v1-K{k}-v{val_band}-lineage`` references
that omit the ``-{plm_short}`` PLM suffix introduced in FARM-EXP.12.
Tests pin:

1. Untagged references in any supported file extension fire.
2. Canonical per-PLM references pass (one case per allowed PLM short).
3. The legacy ``-mini`` smoke dataset is still allowed.
4. Repo-wide scan: the current source tree is clean (regression guard
   for the FARM-EXP.12 rename itself).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LINTER = REPO_ROOT / "scripts" / "lint_dataset_names.py"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LINTER), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_untagged_reference_fires_in_markdown(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("Eval set: bench-v1-K5-v226-lineage on 2026-05-19.\n")
    result = _run([str(target), "--root", str(tmp_path)], cwd=tmp_path)
    assert result.returncode == 1, result.stderr
    assert "bench-v1-K5-v226-lineage" in result.stderr


def test_untagged_reference_fires_in_yaml(tmp_path: Path) -> None:
    target = tmp_path / "cell.yaml"
    target.write_text("eval_set: bench-v1-K10-v226-lineage\n")
    result = _run([str(target), "--root", str(tmp_path)], cwd=tmp_path)
    assert result.returncode == 1, result.stderr


def test_untagged_reference_fires_in_python_string(tmp_path: Path) -> None:
    target = tmp_path / "snippet.py"
    target.write_text('DATASET = "bench-v1-K5-v226-lineage"\n')
    result = _run([str(target), "--root", str(tmp_path)], cwd=tmp_path)
    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize(
    "plm",
    [
        "esm2_150m",
        "esm2_650m",
        "esm2_3b",
        "prot_t5",
        "prostt5",
        "ankh_base",
        "ankh_large",
        "esmc_600m",
        "esmc_300m",
    ],
)
def test_canonical_per_plm_reference_passes(tmp_path: Path, plm: str) -> None:
    target = tmp_path / "ok.md"
    target.write_text(f"Eval set: bench-v1-K5-v226-lineage-{plm} (canonical).\n")
    result = _run([str(target), "--root", str(tmp_path)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_legacy_mini_suffix_passes(tmp_path: Path) -> None:
    target = tmp_path / "smoke.md"
    target.write_text("Smoke dataset: bench-v1-K5-v226-lineage-mini (historical).\n")
    result = _run([str(target), "--root", str(tmp_path)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_repo_tree_is_clean() -> None:
    # Regression guard for the FARM-EXP.12 rename. Runs the linter
    # against its default roots, which mirror the repo layout.
    result = _run([], cwd=REPO_ROOT)
    assert result.returncode == 0, (
        "Untagged dataset references must be rewritten as "
        "bench-v1-K{k}-v226-lineage-{plm_short}:\n" + result.stderr
    )


def test_offence_count_matches_input_lines(tmp_path: Path) -> None:
    target = tmp_path / "two.md"
    target.write_text(
        "line 1 bench-v1-K5-v226-lineage\nline 2 bench-v1-K10-v226-lineage\n"
    )
    result = _run([str(target), "--root", str(tmp_path)], cwd=tmp_path)
    assert result.returncode == 1
    # Two offence lines plus a summary footer.
    body = [
        line for line in result.stderr.splitlines() if "untagged dataset" in line
    ]
    assert len(body) == 2


def test_linter_self_reference_not_flagged() -> None:
    # The linter source contains the literal pattern inside its own
    # docstring. It must not flag itself.
    result = _run(
        [str(LINTER)],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
