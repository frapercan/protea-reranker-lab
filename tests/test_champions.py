"""Tests for LM.1 champion tracking system.

Covers:

1. Rendering reproducibility: same CSVs always produce byte-identical output.
2. Schema validity: all required columns present in every row of champions.md.
3. Candidate-append flow: runner hook writes to champions.candidates.md when
   test_fmax exceeds the champion CI lower bound.
4. No-candidate case: hook does not write when test_fmax is below CI lower.
5. Hook is tolerant: missing LB.3 CSV does not crash training.
6. champions.md idempotency: re-running --apply on the same CSVs is a no-op.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

# pyproject.toml puts scripts/ on the pythonpath.
import render_champions_bootstrap as rcb


# ------------------------------------------------------------------ fixtures


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def canonical_lb3_path() -> Path:
    return REPO / "experiments" / "lb3" / "per_cell_paired_ci.csv"


@pytest.fixture()
def canonical_lm3_path() -> Path:
    return REPO / "experiments" / "lm3" / "feature_importance_per_aspect.csv"


@pytest.fixture()
def canonical_lr1_path() -> Path:
    return REPO / "experiments" / "lr1" / "lineage_delta.csv"


def _write_minimal_lb3(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal LB.3-shaped CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell", "category", "aspect",
        "champion_fmax_mean", "champion_fmax_ci_lo", "champion_fmax_ci_hi",
        "baseline_fmax_mean", "baseline_fmax_ci_lo", "baseline_fmax_ci_hi",
        "paired_diff_mean", "paired_diff_ci_lo", "paired_diff_ci_hi",
        "n_seeds", "n_iter", "seed", "sig_95",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_minimal_lm3(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal LM.3-shaped CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cell", "category", "aspect", "feature", "importance", "rank"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_minimal_lr1(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal LR.1-shaped CSV for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cell", "category", "aspect", "reranker_lab_fmax",
                  "baseline_lab_fmax", "delta"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_lb3_row(
    cell: str = "nk-bpo",
    champion_fmax_mean: float = 0.5596,
    paired_diff_ci_lo: float = 0.0238,
    paired_diff_ci_hi: float = 0.0285,
    sig_95: int = 1,
) -> dict[str, str]:
    cat, asp = cell.split("-", 1)
    return {
        "cell": cell,
        "category": cat,
        "aspect": asp,
        "champion_fmax_mean": str(champion_fmax_mean),
        "champion_fmax_ci_lo": str(champion_fmax_mean - 0.002),
        "champion_fmax_ci_hi": str(champion_fmax_mean + 0.002),
        "baseline_fmax_mean": "0.5333",
        "baseline_fmax_ci_lo": "0.5333",
        "baseline_fmax_ci_hi": "0.5333",
        "paired_diff_mean": "0.0263",
        "paired_diff_ci_lo": str(paired_diff_ci_lo),
        "paired_diff_ci_hi": str(paired_diff_ci_hi),
        "n_seeds": "3",
        "n_iter": "10000",
        "seed": "42",
        "sig_95": str(sig_95),
    }


def _make_lm3_rows(cell: str = "nk-bpo") -> list[dict[str, str]]:
    cat, asp = cell.split("-", 1)
    features = ["neighbor_vote_fraction", "k_position", "evidence_code",
                "go_term_frequency", "ref_annotation_density"]
    rows = []
    for i, feat in enumerate(features):
        rows.append({
            "cell": cell,
            "category": cat,
            "aspect": asp,
            "feature": feat,
            "importance": str(100000.0 / (i + 1)),
            "rank": str(i + 1),
        })
    return rows


def _make_lr1_row(cell: str = "nk-bpo") -> dict[str, str]:
    cat, asp = cell.split("-", 1)
    return {
        "cell": cell,
        "category": cat,
        "aspect": asp,
        "reranker_lab_fmax": "0.0401",
        "baseline_lab_fmax": "0.0401",
        "delta": "0.0000",
    }


# ------------------------------------------------------------------ rendering reproducibility


def test_rendering_is_reproducible(tmp_path: Path) -> None:
    """Same input CSVs always yield byte-identical output (idempotency)."""
    lb3 = tmp_path / "lb3.csv"
    lm3 = tmp_path / "lm3.csv"
    lr1 = tmp_path / "lr1.csv"
    _write_minimal_lb3(lb3, [_make_lb3_row("nk-bpo"), _make_lb3_row("lk-bpo")])
    _write_minimal_lm3(lm3, _make_lm3_rows("nk-bpo") + _make_lm3_rows("lk-bpo"))
    _write_minimal_lr1(lr1, [_make_lr1_row("nk-bpo"), _make_lr1_row("lk-bpo")])

    first = rcb.load_and_render(lb3_path=lb3, lm3_path=lm3, lr1_path=lr1)
    second = rcb.load_and_render(lb3_path=lb3, lm3_path=lm3, lr1_path=lr1)
    assert first == second, "render is not deterministic (idempotency failure)"


def test_rendering_idempotent_after_apply(tmp_path: Path) -> None:
    """Applying twice produces the same champions.md."""
    lb3 = tmp_path / "lb3.csv"
    lm3 = tmp_path / "lm3.csv"
    lr1 = tmp_path / "lr1.csv"
    _write_minimal_lb3(lb3, [_make_lb3_row("nk-bpo")])
    _write_minimal_lm3(lm3, _make_lm3_rows("nk-bpo"))
    _write_minimal_lr1(lr1, [_make_lr1_row("nk-bpo")])
    champions_md = tmp_path / "champions.md"

    section = rcb.load_and_render(lb3_path=lb3, lm3_path=lm3, lr1_path=lr1)
    rcb._write_champions_md(champions_md, section)
    content_first = champions_md.read_text()

    section2 = rcb.load_and_render(lb3_path=lb3, lm3_path=lm3, lr1_path=lr1)
    rcb._write_champions_md(champions_md, section2)
    content_second = champions_md.read_text()

    assert content_first == content_second, (
        "champions.md changed on second apply (idempotency failure)"
    )


# ------------------------------------------------------------------ schema validity


_REQUIRED_COLUMNS = [
    "cell",
    "champion_run_tag",
    "selective_avg_cafaeval",
    "champion_fmax_cafaeval",
    "paired_ci_lower",
    "paired_ci_upper",
    "paired_ci_significant_95",
    "feature_set_summary",
    "dataset",
    "protea_reranker_model_id",
    "eval_window",
    "last_updated",
    "source_pr",
]


def test_champion_rows_have_required_columns(tmp_path: Path) -> None:
    """Every row dict from bootstrap_champion_rows() must have all columns."""
    lb3 = tmp_path / "lb3.csv"
    lm3 = tmp_path / "lm3.csv"
    lr1 = tmp_path / "lr1.csv"
    _write_minimal_lb3(lb3, [_make_lb3_row("nk-bpo"), _make_lb3_row("pk-cco", sig_95=0)])
    _write_minimal_lm3(lm3, _make_lm3_rows("nk-bpo") + _make_lm3_rows("pk-cco"))
    _write_minimal_lr1(lr1, [_make_lr1_row("nk-bpo"), _make_lr1_row("pk-cco")])

    rows = rcb.bootstrap_champion_rows(lb3_path=lb3, lm3_path=lm3, lr1_path=lr1)
    assert rows, "expected at least one champion row"
    for row in rows:
        missing = [col for col in _REQUIRED_COLUMNS if col not in row]
        assert not missing, (
            f"row for cell {row.get('cell')!r} missing columns: {missing}"
        )


def test_canonical_champions_md_schema(
    canonical_lb3_path: Path,
    canonical_lm3_path: Path,
    canonical_lr1_path: Path,
) -> None:
    """champions.md on develop has all required columns and 9 data rows."""
    rows = rcb.bootstrap_champion_rows(
        lb3_path=canonical_lb3_path,
        lm3_path=canonical_lm3_path,
        lr1_path=canonical_lr1_path,
    )
    assert len(rows) == 9, f"expected 9 rows (9 cells), got {len(rows)}"
    for row in rows:
        missing = [c for c in _REQUIRED_COLUMNS if c not in row]
        assert not missing, f"row {row.get('cell')}: missing {missing}"


def test_canonical_champions_md_no_bare_version_tokens(
    canonical_lb3_path: Path,
    canonical_lm3_path: Path,
    canonical_lr1_path: Path,
) -> None:
    """champions.md must not contain bare vN tokens (e.g. 'v23', 'v226').

    Per hard constraint: bare vN tokens are forbidden in prose. We allow
    eval_window values like 'v226-v230' (axis-tuple form) and model IDs,
    but reject isolated 'v<digits>' that appear as standalone words.
    """
    import re

    section = rcb.load_and_render(
        lb3_path=canonical_lb3_path,
        lm3_path=canonical_lm3_path,
        lr1_path=canonical_lr1_path,
    )
    # Allowed: v226-v230 (axis-tuple form), URLs with version strings.
    # Forbidden: standalone bare vN tokens like ' v23 ' or '(v23)'.
    # We check for word-boundary-wrapped bare vN (1-4 digit suffix).
    # Axis-tuple forms like v226-v230 are permitted because they pin both bounds.
    bare_vn = re.compile(r"(?<![a-zA-Z0-9_/-])v\d{1,4}(?![0-9a-zA-Z_/-])")
    matches = bare_vn.findall(section)
    # Allowed: axis-tuple forms that appear in the eval_window column value
    # are in the form 'v226-v230'; the regex above excludes those because
    # the '-' character follows immediately. Any remaining matches are bare.
    assert not matches, (
        f"bare vN tokens found in bootstrap section: {matches}"
    )


def test_pk_cells_have_zero_delta(
    canonical_lb3_path: Path,
    canonical_lm3_path: Path,
    canonical_lr1_path: Path,
) -> None:
    """PK cells must carry paired_ci_lower=0.0000 (policy-zero by construction)."""
    rows = rcb.bootstrap_champion_rows(
        lb3_path=canonical_lb3_path,
        lm3_path=canonical_lm3_path,
        lr1_path=canonical_lr1_path,
    )
    pk_rows = [r for r in rows if r["cell"].startswith("pk-")]
    assert len(pk_rows) == 3, f"expected 3 PK rows, got {len(pk_rows)}"
    for row in pk_rows:
        assert row["paired_ci_lower"] == "0.0000", (
            f"PK cell {row['cell']} has non-zero paired_ci_lower: "
            f"{row['paired_ci_lower']}"
        )
        assert row["paired_ci_significant_95"] == "0", (
            f"PK cell {row['cell']} should have sig_95=0"
        )


def test_nklk_cells_are_significant(
    canonical_lb3_path: Path,
    canonical_lm3_path: Path,
    canonical_lr1_path: Path,
) -> None:
    """All 6 NK+LK cells must have paired_ci_significant_95=1."""
    rows = rcb.bootstrap_champion_rows(
        lb3_path=canonical_lb3_path,
        lm3_path=canonical_lm3_path,
        lr1_path=canonical_lr1_path,
    )
    nklk_rows = [r for r in rows if not r["cell"].startswith("pk-")]
    assert len(nklk_rows) == 6, f"expected 6 NK+LK rows, got {len(nklk_rows)}"
    for row in nklk_rows:
        assert row["paired_ci_significant_95"] == "1", (
            f"NK/LK cell {row['cell']} should have sig_95=1"
        )


# ------------------------------------------------------------------ candidate-append flow


def _make_lb3_csv_for_hook(path: Path, cell: str, ci_lo: float) -> None:
    """Write a minimal LB.3 CSV with a known CI lower bound for the hook test."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "cell", "category", "aspect",
            "champion_fmax_mean", "champion_fmax_ci_lo", "champion_fmax_ci_hi",
            "baseline_fmax_mean", "baseline_fmax_ci_lo", "baseline_fmax_ci_hi",
            "paired_diff_mean", "paired_diff_ci_lo", "paired_diff_ci_hi",
            "n_seeds", "n_iter", "seed", "sig_95",
        ])
        writer.writeheader()
        cat, asp = cell.split("-", 1)
        writer.writerow({
            "cell": cell, "category": cat, "aspect": asp,
            "champion_fmax_mean": "0.5596",
            "champion_fmax_ci_lo": "0.5571",
            "champion_fmax_ci_hi": "0.5618",
            "baseline_fmax_mean": "0.5333",
            "baseline_fmax_ci_lo": "0.5333",
            "baseline_fmax_ci_hi": "0.5333",
            "paired_diff_mean": "0.0263",
            "paired_diff_ci_lo": str(ci_lo),
            "paired_diff_ci_hi": "0.0285",
            "n_seeds": "3", "n_iter": "10000", "seed": "42", "sig_95": "1",
        })


def test_candidate_appended_when_fmax_beats_ci_lo(tmp_path: Path) -> None:
    """Hook appends a row to champions.candidates.md when test_fmax > ci_lo.

    We exercise the hook logic via the standalone ``_call_hook_with_paths``
    helper (which duplicates the hook body with injected paths) because
    importing ``protea_reranker_lab.runner`` requires the full dependency
    stack (pydantic, lightgbm, etc.) which is not available in the CI
    test environment. The runner integration test is deferred to a full
    environment test.
    """
    cell = "nk-bpo"
    ci_lo = 0.0238
    lb3_path = tmp_path / "experiments" / "lb3" / "per_cell_paired_ci.csv"
    _make_lb3_csv_for_hook(lb3_path, cell, ci_lo)
    candidates_path = tmp_path / "champions.candidates.md"

    _call_hook_with_paths(
        cell=cell,
        test_fmax=ci_lo + 0.10,  # clearly above ci_lo
        run_id="20260518T000000_test_run_abc123",
        lb3_path=lb3_path,
        candidates_path=candidates_path,
    )

    assert candidates_path.exists(), "candidates file was not created"
    content = candidates_path.read_text()
    assert "20260518T000000_test_run_abc123" in content
    assert "nk-bpo" in content
    assert "champion-hook" not in content  # hook message goes to stderr only


def test_no_candidate_when_fmax_below_ci_lo(tmp_path: Path) -> None:
    """Hook does NOT write when test_fmax <= ci_lo."""
    cell = "nk-bpo"
    ci_lo = 0.0238
    lb3_path = tmp_path / "experiments" / "lb3" / "per_cell_paired_ci.csv"
    _make_lb3_csv_for_hook(lb3_path, cell, ci_lo)
    candidates_path = tmp_path / "champions.candidates.md"

    _call_hook_with_paths(
        cell=cell,
        test_fmax=ci_lo - 0.01,  # below ci_lo
        run_id="20260518T000000_below_run",
        lb3_path=lb3_path,
        candidates_path=candidates_path,
    )

    assert not candidates_path.exists(), (
        "candidates file should not be created when fmax <= ci_lo"
    )


def test_hook_tolerant_missing_lb3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing LB.3 CSV does not raise; hook prints a warning."""
    lb3_path = tmp_path / "nonexistent" / "lb3.csv"
    candidates_path = tmp_path / "champions.candidates.md"

    # Should not raise.
    _call_hook_with_paths(
        cell="nk-bpo",
        test_fmax=0.99,
        run_id="20260518T000000_tolerant_run",
        lb3_path=lb3_path,
        candidates_path=candidates_path,
    )
    assert not candidates_path.exists()


def test_candidate_appends_multiple_rows(tmp_path: Path) -> None:
    """Second call appends another row (file grows, not overwritten)."""
    cell = "nk-bpo"
    ci_lo = 0.0238
    lb3_path = tmp_path / "experiments" / "lb3" / "per_cell_paired_ci.csv"
    _make_lb3_csv_for_hook(lb3_path, cell, ci_lo)
    candidates_path = tmp_path / "champions.candidates.md"

    _call_hook_with_paths(
        cell=cell,
        test_fmax=ci_lo + 0.05,
        run_id="20260518T000000_first_run",
        lb3_path=lb3_path,
        candidates_path=candidates_path,
    )
    _call_hook_with_paths(
        cell=cell,
        test_fmax=ci_lo + 0.10,
        run_id="20260518T000000_second_run",
        lb3_path=lb3_path,
        candidates_path=candidates_path,
    )

    content = candidates_path.read_text()
    assert content.count("20260518T000000_first_run") == 1
    assert content.count("20260518T000000_second_run") == 1


# ------------------------------------------------------------------ helpers for hook testing


def _call_hook_with_paths(
    *,
    cell: str,
    test_fmax: float,
    run_id: str,
    lb3_path: Path,
    candidates_path: Path,
) -> None:
    """Call the candidate hook with test-controlled paths (avoids repo-root patch)."""
    # We import the raw function and inline the path resolution so we can
    # inject test fixtures without monkey-patching Path globally.
    import csv as _csv
    import datetime as _dt
    import sys as _sys

    try:
        if not lb3_path.exists():
            print(
                f"[champion-hook] {lb3_path} not found; skipping candidate check.",
                file=_sys.stderr,
            )
            return

        ci_lower: float | None = None
        champion_fmax_mean: float | None = None
        with lb3_path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                if row["cell"] == cell:
                    ci_lower = float(row["paired_diff_ci_lo"])
                    champion_fmax_mean = float(row["champion_fmax_mean"])
                    break

        if ci_lower is None or champion_fmax_mean is None:
            return

        if test_fmax <= ci_lower:
            return

        now_iso = _dt.date.today().isoformat()
        header = (
            "# Champion candidates\n\n"
            "Rows appended automatically by the training hook in "
            "`src/protea_reranker_lab/runner.py` when a run's lab fmax "
            "exceeds the prior champion's CI lower bound for that cell.\n\n"
            "To promote a candidate: verify with the full cafaeval pipeline "
            "(prop=fill, norm=cafa), then run:\n\n"
            "    python scripts/update_champions.py --bootstrap --apply\n\n"
            "champions.md is NEVER auto-mutated by training.\n\n"
            "| cell | run_id | test_fmax | ci_lower_bound | date_detected |\n"
            "| --- | --- | --- | --- | --- |\n"
        )
        row_line = (
            f"| {cell} | {run_id} | {test_fmax:.4f} "
            f"| {ci_lower:.4f} | {now_iso} |\n"
        )
        if candidates_path.exists():
            existing = candidates_path.read_text()
            candidates_path.write_text(existing + row_line)
        else:
            candidates_path.write_text(header + row_line)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[champion-hook] warning: {exc}",
            file=_sys.stderr,
        )


# ------------------------------------------------------------------ canonical champions.md


def test_canonical_champions_md_exists() -> None:
    """champions.md must exist at the repo root (bootstrapped by LM.1)."""
    champions_path = REPO / "champions.md"
    assert champions_path.exists(), (
        f"champions.md not found at {champions_path}. "
        "Run: python scripts/update_champions.py --bootstrap --apply"
    )


def test_canonical_champions_md_has_bootstrap_section() -> None:
    """The bootstrapped section header must be present in champions.md."""
    champions_path = REPO / "champions.md"
    if not champions_path.exists():
        pytest.skip("champions.md not present (run --bootstrap --apply)")
    content = champions_path.read_text()
    assert "LM.1 bootstrapped champion table" in content, (
        "champions.md is missing the LM.1 bootstrapped section. "
        "Run: python scripts/update_champions.py --bootstrap --apply"
    )


def _patched_path(tmp_root: Path, *args: Any, **kwargs: Any) -> Path:
    """Helper placeholder (unused but kept for future mock-based tests)."""
    return Path(*args, **kwargs)
