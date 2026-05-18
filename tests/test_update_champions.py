"""Tests for FARM-EXP.4 scripts/update_champions.py.

Covers:

1. Promotion when candidate CI lo > prior mean.
2. No promotion when candidate CI lo <= prior mean.
3. Independent tracking across triples (eval_set / tier / aspect).
4. Dry-run does not write champions.md.
5. Symlinks are RELATIVE (`os.readlink` starts with ``..``).
6. ``--explain`` prints each candidate's CI bounds and verdict.
7. The loader is the FARM-EXP.3 ``bootstrap_cis.RunRecord.from_dict``
   path (reuse, not re-implementation).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# pyproject puts scripts/ on the pythonpath.
import bootstrap_cis as bc
import update_champions as uc


# ----------------------------------------------------- fixture builder


def _write_run(
    out_dir: Path,
    *,
    run_id: str,
    samples: np.ndarray,
    cell: str = "nk-bpo",
    eval_set: str = "bench-v1-K5-filtered",
    shortid: str | None = None,
    extra_axis: dict[str, Any] | None = None,
) -> Path:
    """Persist a FARM-EXP.3-shaped run record JSON.

    The lab convention is one run per directory; we mirror that so
    ``_infer_run_dir`` resolves to the right path.
    """
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    axis: dict[str, Any] = {
        "plm": "esm2_3b",
        "k": 5,
        "reranker": "lgbm.per_cell_9",
        "features": "v6",
        "eval_set": eval_set,
        "propagation": "tpr_pred",
        "ensemble": "none",
    }
    if extra_axis:
        axis.update(extra_axis)
    payload = {
        "run_id": run_id,
        "axis": axis,
        "shortid": shortid or f"{run_id[:12]:0<12}",
        "cell": cell,
        "fmax_samples": samples.tolist(),
        "fmax_point": float(samples.mean()),
    }
    json_path = run_dir / "run.json"
    json_path.write_text(json.dumps(payload))
    return json_path


def _normal_samples(
    mean: float, std: float, *, size: int = 200, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=size)


# ----------------------------------------------------- promotion rule


def test_promotion_when_ci_lo_beats_prior_mean(tmp_path: Path) -> None:
    """B's CI lo (~0.46) exceeds A's mean (0.40) -> B becomes champion."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_A_run",
        samples=_normal_samples(0.40, 0.02, seed=1),
    )
    _write_run(
        runs,
        run_id="20260102T000000_B_run",
        samples=_normal_samples(0.50, 0.02, seed=2),
    )
    winners, traces, _ = uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    triple = uc.TriplKey(
        eval_set="bench-v1-K5-filtered", tier="nk", aspect="bpo"
    )
    assert triple in winners
    assert winners[triple].record.run_id == "20260102T000000_B_run"
    verdicts = [d.verdict for d in traces[triple]]
    assert verdicts == ["seeded", "promoted"]


def test_no_promotion_when_ci_lo_below_prior_mean(
    tmp_path: Path,
) -> None:
    """B's CI lo (~0.44) does NOT exceed A's mean (0.50) -> A stays."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_A_run",
        samples=_normal_samples(0.50, 0.02, seed=10),
    )
    _write_run(
        runs,
        run_id="20260102T000000_B_run",
        samples=_normal_samples(0.48, 0.02, seed=11),
    )
    winners, traces, _ = uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    triple = uc.TriplKey(
        eval_set="bench-v1-K5-filtered", tier="nk", aspect="bpo"
    )
    assert triple in winners
    assert winners[triple].record.run_id == "20260101T000000_A_run"
    verdicts = [d.verdict for d in traces[triple]]
    assert verdicts == ["seeded", "rejected"]


# ----------------------------------------------------- per-triple isolation


def test_separate_triples_track_independently(
    tmp_path: Path,
) -> None:
    """Different (eval_set, tier, aspect) triples never compete."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_nk_bpo_v1",
        samples=_normal_samples(0.40, 0.02, seed=20),
        cell="nk-bpo",
        eval_set="bench-v1-K5-filtered",
    )
    _write_run(
        runs,
        run_id="20260102T000000_pk_cco_v1",
        samples=_normal_samples(0.70, 0.02, seed=21),
        cell="pk-cco",
        eval_set="bench-v1-K5-v226-lineage",
    )
    winners, _, _ = uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    assert len(winners) == 2
    nk_bpo = uc.TriplKey(
        eval_set="bench-v1-K5-filtered", tier="nk", aspect="bpo"
    )
    pk_cco = uc.TriplKey(
        eval_set="bench-v1-K5-v226-lineage",
        tier="pk",
        aspect="cco",
    )
    assert nk_bpo in winners and pk_cco in winners


# ----------------------------------------------------- dry-run vs apply


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    """``--apply`` off must not touch the filesystem."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_only_run",
        samples=_normal_samples(0.40, 0.02, seed=30),
    )
    champions_md = tmp_path / "champions.md"
    sym_root = tmp_path / "runs" / "champion"
    uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=sym_root,
        apply=False,
    )
    assert not champions_md.exists()
    assert not sym_root.exists()


def test_apply_writes_champions_md(tmp_path: Path) -> None:
    """``--apply`` writes a non-empty markdown table."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_only_run",
        samples=_normal_samples(0.40, 0.02, seed=40),
    )
    champions_md = tmp_path / "champions.md"
    uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    body = champions_md.read_text()
    assert "# Champions" in body
    assert "bench-v1-K5-filtered" in body
    assert "nk" in body and "bpo" in body
    assert "20260101T000000_only_run" in body


# ----------------------------------------------------- symlink policy


def test_symlink_is_relative(tmp_path: Path) -> None:
    """``os.readlink`` returns a path that starts with `..`."""
    runs = tmp_path / "runs"
    json_path = _write_run(
        runs,
        run_id="20260101T000000_only_run",
        samples=_normal_samples(0.40, 0.02, seed=50),
    )
    sym_root = tmp_path / "runs" / "champion"
    uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=sym_root,
        apply=True,
    )
    link = (
        sym_root
        / "bench-v1-K5-filtered"
        / "nk"
        / "bpo"
    )
    assert link.is_symlink()
    target = os.readlink(link)
    assert target.startswith(".."), (
        f"expected relative target, got {target!r}"
    )
    # Resolves to the run dir (where the run.json lives).
    assert (link.parent / target).resolve() == json_path.parent.resolve()


def test_apply_refreshes_existing_symlink(tmp_path: Path) -> None:
    """A second run that beats the prior champion repoints the symlink."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_A_run",
        samples=_normal_samples(0.40, 0.02, seed=60),
    )
    sym_root = tmp_path / "runs" / "champion"
    uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=sym_root,
        apply=True,
    )
    # Now add a winner.
    _write_run(
        runs,
        run_id="20260102T000000_B_run",
        samples=_normal_samples(0.55, 0.02, seed=61),
    )
    uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=sym_root,
        apply=True,
    )
    link = (
        sym_root
        / "bench-v1-K5-filtered"
        / "nk"
        / "bpo"
    )
    target = os.readlink(link)
    assert "20260102T000000_B_run" in target


# ----------------------------------------------------- explain CLI


def test_explain_prints_decision_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--explain`` emits the candidate's CI bounds + verdict."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_A_run",
        samples=_normal_samples(0.40, 0.02, seed=70),
    )
    _write_run(
        runs,
        run_id="20260102T000000_B_run",
        samples=_normal_samples(0.50, 0.02, seed=71),
    )
    rc = uc.main(
        [
            "--runs-dir",
            str(runs),
            "--explain",
            "bench-v1-K5-filtered:nk:bpo",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "Explain: bench-v1-K5-filtered:nk:bpo" in captured.out
    assert "seeded" in captured.out
    assert "promoted" in captured.out
    assert "ci=[" in captured.out


def test_explain_unknown_triple_prints_no_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    rc = uc.main(
        [
            "--runs-dir",
            str(runs),
            "--explain",
            "bench-v1-K5-filtered:nk:bpo",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "no runs in this triple" in out


def test_explain_rejects_malformed_triple(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    rc = uc.main(
        ["--runs-dir", str(runs), "--explain", "not-a-triple"]
    )
    assert rc == 2


# ----------------------------------------------------- loader reuse


def test_loader_is_bootstrap_cis_runrecord(tmp_path: Path) -> None:
    """Champion tracker MUST use bootstrap_cis.RunRecord; no fork."""
    runs = tmp_path / "runs"
    json_path = _write_run(
        runs,
        run_id="20260101T000000_only_run",
        samples=_normal_samples(0.40, 0.02, seed=80),
    )
    pairs = uc._load_records_with_source(runs)
    assert len(pairs) == 1
    record, source = pairs[0]
    assert isinstance(record, bc.RunRecord)
    assert source == json_path


def test_loader_skips_records_missing_fmax_samples(
    tmp_path: Path,
) -> None:
    """Legacy run.json without FARM-EXP.3 fields is silently skipped."""
    runs = tmp_path / "runs"
    runs.mkdir()
    legacy = runs / "legacy"
    legacy.mkdir()
    (legacy / "run.json").write_text(
        json.dumps(
            {
                "run_id": "legacy-1",
                "metrics": {"test_fmax": 0.123},
            }
        )
    )
    pairs = uc._load_records_with_source(runs)
    assert pairs == []


# ----------------------------------------------------- range-pinning guard


def test_record_without_eval_set_axis_is_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Champions must pin a validation range (v18 mistake)."""
    runs = tmp_path / "runs"
    # Build a record whose axis lacks eval_set.
    run_dir = runs / "no_range"
    run_dir.mkdir(parents=True)
    samples = _normal_samples(0.40, 0.02, seed=90)
    payload = {
        "run_id": "no-range-run",
        "axis": {
            "plm": "esm2_3b",
            "k": 5,
            "reranker": "lgbm.per_cell_9",
            "features": "v6",
            # eval_set deliberately omitted
            "propagation": "tpr_pred",
            "ensemble": "none",
        },
        "shortid": "deadbeef0000",
        "cell": "nk-bpo",
        "fmax_samples": samples.tolist(),
    }
    (run_dir / "run.json").write_text(json.dumps(payload))

    winners, _, _ = uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=tmp_path / "runs" / "champion",
        apply=False,
    )
    assert winners == {}
    captured = capsys.readouterr()
    assert "axis.eval_set" in captured.err


def test_record_with_malformed_cell_is_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "bad_cell"
    run_dir.mkdir(parents=True)
    samples = _normal_samples(0.40, 0.02, seed=91)
    payload = {
        "run_id": "bad-cell-run",
        "axis": {
            "plm": "esm2_3b",
            "k": 5,
            "reranker": "lgbm.per_cell_9",
            "features": "v6",
            "eval_set": "bench-v1-K5-filtered",
            "propagation": "tpr_pred",
            "ensemble": "none",
        },
        "cell": "not_a_dashed_cell",
        "fmax_samples": samples.tolist(),
    }
    (run_dir / "run.json").write_text(json.dumps(payload))
    winners, _, _ = uc.run(
        runs_dir=runs,
        champions_file=tmp_path / "champions.md",
        symlink_root=tmp_path / "runs" / "champion",
        apply=False,
    )
    assert winners == {}
    captured = capsys.readouterr()
    assert "malformed cell" in captured.err


# ----------------------------------------------------- empty scaffold


def test_missing_runs_dir_is_treated_as_empty(
    tmp_path: Path,
) -> None:
    """A nonexistent runs/ (gitignored, fresh clone) yields a scaffold."""
    runs = tmp_path / "does_not_exist"
    champions_md = tmp_path / "champions.md"
    winners, _, _ = uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    assert winners == {}
    assert champions_md.exists()
    assert "# Champions" in champions_md.read_text()


def test_render_scaffold_when_no_records(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    champions_md = tmp_path / "champions.md"
    uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    body = champions_md.read_text()
    assert "# Champions" in body
    assert "no farm-exp.3-format run records" in body.lower()


# ----------------------------------------------------- manual appendix


def test_manual_appendix_preserved_across_apply(tmp_path: Path) -> None:
    """A pre-FARM-EXP.3 manual entry survives ``--apply`` re-renders.

    The marker pair ``<!-- MANUAL_ENTRIES_BEGIN -->`` ...
    ``<!-- MANUAL_ENTRIES_END -->`` delimits a manually-curated
    appendix that records champions for which no FARM-EXP.3 run.json
    exists yet (writer slice deferred to FARM-EXP.5+). This is the
    LR.4 acceptance contract: the leakage-free selective rerank
    champion entry must not be silently dropped when the auto-walker
    rewrites the file.
    """
    # Seed an existing champions.md with a manual entry.
    champions_md = tmp_path / "champions.md"
    initial = (
        uc.CHAMPIONS_HEADER
        + "_scaffold body_\n"
        + "\n"
        + uc.MANUAL_ENTRIES_BEGIN
        + "\n"
        + "## Manual entries (pre-FARM-EXP.3 records)\n"
        + "\n"
        + "LR.4: leakage-free selective rerank entry.\n"
        + uc.MANUAL_ENTRIES_END
        + "\n"
    )
    champions_md.write_text(initial)
    runs = tmp_path / "runs"
    runs.mkdir()
    uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    rewritten = champions_md.read_text()
    assert uc.MANUAL_ENTRIES_BEGIN in rewritten
    assert uc.MANUAL_ENTRIES_END in rewritten
    assert "LR.4: leakage-free selective rerank entry." in rewritten


def test_manual_appendix_absent_marker_pair_yields_no_appendix(
    tmp_path: Path,
) -> None:
    """A file without the markers contributes no appendix to the re-render."""
    champions_md = tmp_path / "champions.md"
    champions_md.write_text(
        uc.CHAMPIONS_HEADER + "_scaffold body, no markers_\n"
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    rewritten = champions_md.read_text()
    assert uc.MANUAL_ENTRIES_BEGIN not in rewritten


def test_manual_appendix_preserved_with_real_winners(tmp_path: Path) -> None:
    """Appendix survives the rich-content branch of render_champions_md."""
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="20260101T000000_only_run",
        samples=_normal_samples(0.40, 0.02, seed=40),
    )
    champions_md = tmp_path / "champions.md"
    # Pre-seed the file with a champion table + manual appendix; the
    # next --apply run must regenerate the table and re-append the
    # appendix verbatim.
    pre = (
        uc.CHAMPIONS_HEADER
        + "(stale body)\n\n"
        + uc.MANUAL_ENTRIES_BEGIN
        + "\nLR.4 entry kept.\n"
        + uc.MANUAL_ENTRIES_END
        + "\n"
    )
    champions_md.write_text(pre)
    uc.run(
        runs_dir=runs,
        champions_file=champions_md,
        symlink_root=tmp_path / "runs" / "champion",
        apply=True,
    )
    body = champions_md.read_text()
    assert "20260101T000000_only_run" in body  # rich-content branch fired
    assert "LR.4 entry kept." in body
    assert "(stale body)" not in body


def test_extract_manual_appendix_handles_missing_markers() -> None:
    """The extractor returns empty when either marker is absent."""
    assert uc.extract_manual_appendix("# Champions\nbody\n") == ""
    assert uc.extract_manual_appendix(
        "# Champions\n" + uc.MANUAL_ENTRIES_BEGIN + "\nstuff\n"
    ) == ""


# ----------------------------------------------------- TriplKey parse


def test_triple_parse_roundtrip() -> None:
    raw = "bench-v1-K5-filtered:nk:bpo"
    parsed = uc.TriplKey.parse(raw)
    assert parsed.as_str() == raw


def test_triple_parse_rejects_empty_component() -> None:
    with pytest.raises(ValueError):
        uc.TriplKey.parse("bench-v1-K5-filtered::bpo")


def test_triple_parse_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError):
        uc.TriplKey.parse("nk:bpo")
