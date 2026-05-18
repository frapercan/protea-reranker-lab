"""Tests for the FARM-EXP.8 constrained-grid orchestrator.

The orchestrator is a dispatch shell: catalog -> per-cell plans ->
``scripts/run.py`` subprocesses, with a champion-update hook after
every batch. Tests pin the slice acceptance:

1. Reads ``experiments/_catalog/transversal.yaml`` (FARM-EXP.2 schema).
2. Schedules cells in batches of 10 (size configurable).
3. Writes per-cell ``run.json`` upfront with axis tuple + shortid +
   git sha + resolved hparams + artefact paths.
4. Champion auto-updater fires after every batch (FARM-EXP.4).
5. Idempotent re-runs (status=ok skipped).
6. Per-cell artefacts land under ``runs/transversal/<shortid>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import run_transversal_grid as rtg


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def fake_catalog(tmp_path: Path) -> Path:
    stanzas = [
        {
            "shortid": "aaaaaaaaaaaa",
            "status": "planned",
            "plm": "esmc_300m",
            "k": 5,
            "reranker": "lgbm.per_cell_9",
            "features": "v6+lineage-leakfree",
            "eval_set": "bench-v1-K5-v226-lineage",
            "propagation": "tpr_pred",
            "ensemble": "none",
        },
        {
            "shortid": "bbbbbbbbbbbb",
            "status": "planned",
            "plm": "ankh_base",
            "k": 10,
            "reranker": "lgbm.per_tier_3",
            "features": "v6",
            "eval_set": "bench-v1-K5-filtered",
            "propagation": "tpr_pred",
            "ensemble": "none",
        },
        {
            "shortid": "cccccccccccc",
            "status": "planned",
            "plm": "esm2_3b",
            "k": 5,
            "reranker": "alignment_weighted",
            "features": "v6",
            "eval_set": "bench-v1-K5-filtered",
            "propagation": "tpr_pred",
            "ensemble": "none",
        },
    ]
    path = tmp_path / "transversal.yaml"
    path.write_text(yaml.safe_dump(
        {"schema_version": "v1", "cells": stanzas}, sort_keys=False
    ))
    return path


@pytest.fixture
def fake_spec_root(tmp_path: Path) -> Path:
    return tmp_path / "specs"


@pytest.fixture
def fake_runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _write_spec(
    spec_root: Path, shortid: str, cell: str, seed: int = 42,
) -> Path:
    out = spec_root / shortid / f"{cell}_seed{seed}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": f"{shortid}_{cell}_seed{seed}",
        "dataset": {"manifest": "datasets/bench-v1-K5-filtered/manifest.json"},
        "model": {
            "kind": "lgbm_reranker",
            "defaults": {
                "objective": "lambdarank",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 100,
                "num_boost_round": 10000,
                "early_stopping_rounds": 100,
            },
        },
        "training": {
            "cell": cell,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": seed,
        },
        "sweep": {"backend": "none"},
        "output_dir": f"runs/transversal/{shortid}/{cell}/seed{seed}",
        "tags": ["transversal", cell],
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out


def _ns(**overrides: Any) -> argparse.Namespace:
    """Minimal argparse.Namespace with orchestrator defaults applied."""
    defaults: dict[str, Any] = {
        "catalog": rtg.DEFAULT_CATALOG,
        "spec_root": rtg.DEFAULT_SPEC_ROOT,
        "runs_root": rtg.DEFAULT_RUNS_ROOT,
        "batch_size": rtg.DEFAULT_BATCH_SIZE,
        "cells": ",".join(rtg.DEFAULT_CELLS),
        "seeds": None,
        "plms": None,
        "features": None,
        "eval_sets": None,
        "rerankers": None,
        "shortids": None,
        "limit": None,
        "dry_run": False,
        "no_champion_update": True,  # tests opt out of champion sweeps
        "python": None,
        "run_script": rtg.REPO / "scripts" / "run.py",
        "champion_script": rtg.REPO / "scripts" / "update_champions.py",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------- catalog


def test_load_catalog_returns_cells(fake_catalog: Path) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    assert len(stanzas) == 3
    assert {c["shortid"] for c in stanzas} == {
        "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc",
    }


def test_load_catalog_rejects_missing_cells(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: v1\n")
    with pytest.raises(ValueError, match="cells"):
        rtg._load_catalog(bad)


def test_load_real_catalog_is_parseable() -> None:
    """The committed FARM-EXP.2 catalog must parse cleanly."""
    if not rtg.DEFAULT_CATALOG.exists():
        pytest.skip("default catalog not materialised in this worktree")
    stanzas = rtg._load_catalog(rtg.DEFAULT_CATALOG)
    assert len(stanzas) >= 100, "catalog should hold the full cell set"
    for stanza in stanzas:
        assert "shortid" in stanza
        for key in rtg._AXIS_KEYS:
            assert key in stanza, f"missing axis key {key} in {stanza}"


# ---------------------------------------------------------------- filtering


def test_filter_by_features(fake_catalog: Path) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    kept = rtg._filter_stanzas(
        stanzas,
        plms=None, features=("v6",), eval_sets=None,
        rerankers=None, shortids=None,
    )
    assert [s["shortid"] for s in kept] == ["bbbbbbbbbbbb", "cccccccccccc"]


def test_filter_by_eval_set(fake_catalog: Path) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    kept = rtg._filter_stanzas(
        stanzas,
        plms=None, features=None,
        eval_sets=("bench-v1-K5-v226-lineage",),
        rerankers=None, shortids=None,
    )
    assert [s["shortid"] for s in kept] == ["aaaaaaaaaaaa"]


def test_filter_by_shortid(fake_catalog: Path) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    kept = rtg._filter_stanzas(
        stanzas, plms=None, features=None, eval_sets=None,
        rerankers=None, shortids=("bbbbbbbbbbbb",),
    )
    assert [s["shortid"] for s in kept] == ["bbbbbbbbbbbb"]


def test_filter_empty_returns_all(fake_catalog: Path) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    kept = rtg._filter_stanzas(
        stanzas, plms=None, features=None, eval_sets=None,
        rerankers=None, shortids=None,
    )
    assert kept == stanzas


# ---------------------------------------------------------------- planning


def test_plan_jobs_cartesian_count(
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    plans = rtg._plan_jobs(
        stanzas,
        cells=("nk-bpo", "lk-cco"),
        seeds=(42, 7),
        spec_root=fake_spec_root,
        runs_root=fake_runs_root,
    )
    # 3 stanzas x 2 cells x 2 seeds.
    assert len(plans) == 12
    keys = {p.key for p in plans}
    assert (
        rtg.JobKey("aaaaaaaaaaaa", "nk-bpo", 42) in keys
    )
    assert (
        rtg.JobKey("cccccccccccc", "lk-cco", 7) in keys
    )


def test_plan_jobs_resolves_paths(
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    plans = rtg._plan_jobs(
        stanzas[:1],
        cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )
    plan = plans[0]
    assert plan.spec_path == (
        fake_spec_root / "aaaaaaaaaaaa" / "nk-bpo_seed42.yaml"
    )
    assert plan.output_dir == (
        fake_runs_root / "aaaaaaaaaaaa" / "nk-bpo" / "seed42"
    )
    assert plan.run_json_path == plan.output_dir / "run.json"


def test_plan_marks_spec_existence(
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    # Materialise only one spec.
    _write_spec(fake_spec_root, "aaaaaaaaaaaa", "nk-bpo")
    plans = rtg._plan_jobs(
        stanzas, cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )
    by_short = {p.key.shortid: p for p in plans}
    assert by_short["aaaaaaaaaaaa"].spec_exists is True
    assert by_short["bbbbbbbbbbbb"].spec_exists is False
    assert by_short["cccccccccccc"].spec_exists is False


# ---------------------------------------------------------------- batching


def test_batches_split_in_groups_of_n() -> None:
    items = list(range(25))
    batches = list(rtg._batches(items, 10))
    assert [len(b) for b in batches] == [10, 10, 5]


def test_batches_default_size_is_ten() -> None:
    assert rtg.DEFAULT_BATCH_SIZE == 10


def test_batches_rejects_zero() -> None:
    with pytest.raises(ValueError):
        list(rtg._batches([1, 2, 3], 0))


# ---------------------------------------------------------------- upfront run.json


def test_upfront_run_json_carries_axis_and_shortid(
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    _write_spec(fake_spec_root, "aaaaaaaaaaaa", "nk-bpo")
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]
    rtg._write_upfront_run_json(plan, run_id="rid-test", status="pending")
    payload = json.loads(plan.run_json_path.read_text())
    assert payload["run_id"] == "rid-test"
    assert payload["status"] == "pending"
    assert payload["shortid"] == "aaaaaaaaaaaa"
    assert payload["cell"] == "nk-bpo"
    assert payload["tier"] == "nk"
    assert payload["aspect"] == "bpo"
    assert payload["seed"] == 42
    assert payload["axis"]["plm"] == "esmc_300m"
    assert payload["axis"]["eval_set"] == "bench-v1-K5-v226-lineage"
    assert "git_sha" in payload
    assert payload["resolved_hparams"]["model"]["learning_rate"] == 0.05
    assert payload["artefact_paths"]["model"].endswith("model.txt")
    assert payload["artefact_paths"]["predictions"].endswith(
        "predictions.parquet"
    )


def test_upfront_run_json_blocked_records_reason(
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]
    # No spec written; output_dir gets created by writer.
    rtg._write_upfront_run_json(
        plan, run_id="rid-blk", status="blocked",
        reason="spec_not_materialized",
    )
    payload = json.loads(plan.run_json_path.read_text())
    assert payload["status"] == "blocked"
    assert payload["reason"] == "spec_not_materialized"
    assert payload["resolved_hparams"] == {}


# ---------------------------------------------------------------- idempotency


def test_is_complete_true_on_status_ok(tmp_path: Path) -> None:
    rj = tmp_path / "run.json"
    rj.write_text(json.dumps({"status": "ok"}))
    assert rtg._is_complete(rj) is True


@pytest.mark.parametrize("status", ["pending", "failed", "blocked"])
def test_is_complete_false_on_non_ok(tmp_path: Path, status: str) -> None:
    rj = tmp_path / "run.json"
    rj.write_text(json.dumps({"status": status}))
    assert rtg._is_complete(rj) is False


def test_is_complete_false_when_missing(tmp_path: Path) -> None:
    assert rtg._is_complete(tmp_path / "absent.json") is False


def test_dispatch_skips_completed(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    """A plan whose run.json says ok must be skipped without subprocess."""
    stanzas = rtg._load_catalog(fake_catalog)
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    plan.run_json_path.write_text(json.dumps({"status": "ok"}))
    # Refresh plan completion state to mimic the planner's snapshot.
    plan.completed = rtg._is_complete(plan.run_json_path)

    calls: list[Any] = []

    def fail_subprocess(*args: Any, **kw: Any) -> Any:
        calls.append(args)
        raise AssertionError("subprocess must not run for completed plans")

    monkeypatch.setattr(rtg.subprocess, "run", fail_subprocess)
    result = rtg._dispatch_batch(
        [plan], python_exec="python", run_py=Path("run.py"), dry_run=False,
    )
    assert plan.key in result.skipped
    assert not calls


# ---------------------------------------------------------------- dispatch


def test_dispatch_blocks_when_spec_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]

    def fail_subprocess(*args: Any, **kw: Any) -> Any:
        raise AssertionError("subprocess must not run for blocked plans")

    monkeypatch.setattr(rtg.subprocess, "run", fail_subprocess)
    monkeypatch.setattr(rtg, "_git_sha", lambda repo=rtg.REPO: "deadbeef")
    result = rtg._dispatch_batch(
        [plan], python_exec="python", run_py=Path("run.py"), dry_run=False,
    )
    assert plan.key in result.blocked
    payload = json.loads(plan.run_json_path.read_text())
    assert payload["status"] == "blocked"
    assert payload["reason"] == "spec_not_materialized"


def test_dispatch_invokes_run_py(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    spec_path = _write_spec(fake_spec_root, "aaaaaaaaaaaa", "nk-bpo")
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]

    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, returncode: int = 0, **_: Any) -> None:
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd: list[str], **kw: Any) -> _Proc:
        calls.append(cmd)
        # Simulate scripts/run.py writing run.json with status=ok.
        plan.run_json_path.parent.mkdir(parents=True, exist_ok=True)
        plan.run_json_path.write_text(json.dumps({"status": "ok"}))
        return _Proc(returncode=0)

    monkeypatch.setattr(rtg.subprocess, "run", fake_run)
    monkeypatch.setattr(rtg, "_git_sha", lambda repo=rtg.REPO: "deadbeef")
    result = rtg._dispatch_batch(
        [plan], python_exec="python3",
        run_py=Path("/tmp/run.py"), dry_run=False,
    )
    assert plan.key in result.ok
    run_calls = [c for c in calls if c and c[0] == "python3"]
    assert run_calls, f"expected python3 invocation, got {calls}"
    assert run_calls[0][1] == "/tmp/run.py"
    assert run_calls[0][2] == str(spec_path)


def test_dispatch_marks_failure_on_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    _write_spec(fake_spec_root, "aaaaaaaaaaaa", "nk-bpo")
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]

    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(
        rtg.subprocess, "run", lambda *a, **k: _Proc(),
    )
    monkeypatch.setattr(rtg, "_git_sha", lambda repo=rtg.REPO: "deadbeef")
    result = rtg._dispatch_batch(
        [plan], python_exec="python",
        run_py=Path("/tmp/run.py"), dry_run=False,
    )
    assert plan.key in result.failed


def test_dispatch_dry_run_does_not_touch_disk_or_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    _write_spec(fake_spec_root, "aaaaaaaaaaaa", "nk-bpo")
    plan = rtg._plan_jobs(
        stanzas[:1], cells=("nk-bpo",), seeds=(42,),
        spec_root=fake_spec_root, runs_root=fake_runs_root,
    )[0]

    def fail_run(*args: Any, **kw: Any) -> Any:
        raise AssertionError("dry-run must not subprocess")

    monkeypatch.setattr(rtg.subprocess, "run", fail_run)
    result = rtg._dispatch_batch(
        [plan], python_exec="python",
        run_py=Path("/tmp/run.py"), dry_run=True,
    )
    assert plan.key in result.attempted
    assert not plan.run_json_path.exists()


# ---------------------------------------------------------------- champion


def test_champion_update_invoked_after_batch(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    """`run(...)` must call update_champions.py once per non-dry batch."""
    stanzas = rtg._load_catalog(fake_catalog)
    for st in stanzas:
        _write_spec(fake_spec_root, st["shortid"], "nk-bpo")

    calls: list[list[str]] = []

    class _Proc:
        def __init__(self, returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd: list[str], **kw: Any) -> _Proc:
        calls.append(list(cmd))
        # Pretend run.py succeeded by writing run.json status=ok.
        if cmd[1].endswith("run.py"):
            target = Path(cmd[2])
            payload = yaml.safe_load(target.read_text())
            out_dir = fake_runs_root / Path(payload["output_dir"]).relative_to(
                Path("runs/transversal")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run.json").write_text(json.dumps({"status": "ok"}))
        return _Proc(returncode=0)

    monkeypatch.setattr(rtg.subprocess, "run", fake_run)
    # Use a fake git_sha to avoid the real `git` lookup.
    monkeypatch.setattr(rtg, "_git_sha", lambda repo=None: "deadbeef")
    args = _ns(
        catalog=fake_catalog,
        spec_root=fake_spec_root,
        runs_root=fake_runs_root,
        cells="nk-bpo",
        seeds=None,
        batch_size=2,
        no_champion_update=False,
        run_script=Path("/tmp/run.py"),
        champion_script=Path("/tmp/update_champions.py"),
    )
    rc = rtg.run(args)
    assert rc == 0
    champion_calls = [c for c in calls if c[1].endswith("update_champions.py")]
    # 3 planned jobs / batch_size=2 -> 2 batches -> 2 champion invocations.
    assert len(champion_calls) == 2
    for cc in champion_calls:
        assert "--apply" in cc
        assert "--runs-dir" in cc


def test_champion_update_skipped_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
    fake_catalog: Path, fake_spec_root: Path, fake_runs_root: Path,
) -> None:
    stanzas = rtg._load_catalog(fake_catalog)
    _write_spec(fake_spec_root, stanzas[0]["shortid"], "nk-bpo")

    calls: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd: list[str], **kw: Any) -> _Proc:
        calls.append(list(cmd))
        if cmd[1].endswith("run.py"):
            target = Path(cmd[2])
            payload = yaml.safe_load(target.read_text())
            out_dir = fake_runs_root / Path(payload["output_dir"]).relative_to(
                Path("runs/transversal")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run.json").write_text(json.dumps({"status": "ok"}))
        return _Proc()

    monkeypatch.setattr(rtg.subprocess, "run", fake_run)
    monkeypatch.setattr(rtg, "_git_sha", lambda repo=None: "deadbeef")
    args = _ns(
        catalog=fake_catalog,
        spec_root=fake_spec_root,
        runs_root=fake_runs_root,
        cells="nk-bpo",
        shortids=stanzas[0]["shortid"],
        batch_size=10,
        no_champion_update=True,
        run_script=Path("/tmp/run.py"),
        champion_script=Path("/tmp/update_champions.py"),
    )
    rc = rtg.run(args)
    assert rc == 0
    assert not any(c[1].endswith("update_champions.py") for c in calls)


# ---------------------------------------------------------------- CLI parse


def test_parse_seeds_default_is_42() -> None:
    assert rtg._parse_seeds(None) == (42,)


def test_parse_seeds_csv() -> None:
    assert rtg._parse_seeds("42,7,137") == (42, 7, 137)


def test_parse_seeds_bad_token_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        rtg._parse_seeds("42,oops")


def test_parse_csv_empty_returns_empty_list() -> None:
    assert rtg._parse_csv("") == []
    assert rtg._parse_csv(None) == []


def test_parser_default_batch_size_matches_constant() -> None:
    parser = rtg._build_parser()
    args = parser.parse_args([])
    assert args.batch_size == rtg.DEFAULT_BATCH_SIZE


# ---------------------------------------------------------------- summary


def test_summary_aggregates_results() -> None:
    plans = [
        rtg.JobPlan(
            key=rtg.JobKey("sid", "nk-bpo", 42),
            stanza={}, spec_path=Path("a"), output_dir=Path("b"),
            run_json_path=Path("b/run.json"),
            spec_exists=True, completed=False,
        ),
    ]
    r1 = rtg.BatchResult(
        attempted=[rtg.JobKey("sid", "nk-bpo", 42)],
        ok=[rtg.JobKey("sid", "nk-bpo", 42)],
    )
    r2 = rtg.BatchResult(
        blocked=[rtg.JobKey("sid", "lk-cco", 42)],
    )
    txt = rtg._summary(plans, [r1, r2])
    assert "ok=1" in txt
    assert "blocked=1" in txt
