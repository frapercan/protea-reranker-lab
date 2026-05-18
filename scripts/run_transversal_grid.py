#!/usr/bin/env python
"""Constrained transversal grid orchestrator (FARM-EXP.8).

Reads ``experiments/_catalog/transversal.yaml`` (FARM-EXP.2), expands
each axis-tuple stanza into per-cell training jobs (one per
``tier-aspect`` x seed), batches them in groups of 10, and dispatches
each batch sequentially through ``scripts/run.py``. After every batch
the FARM-EXP.4 champion auto-updater is invoked (``--apply``) so the
champion table stays current as the sweep progresses.

The orchestrator is idempotent: a job whose ``run.json`` already
reports ``status=ok`` is skipped on the next invocation. Re-runs of
the same catalog are therefore a no-op once complete.

Spec materialisation policy
---------------------------

Each ``(shortid, cell, seed)`` triple resolves to an ``ExperimentSpec``
YAML at::

    experiments/_generated/transversal/<shortid>/<cell>_seed<seed>.yaml

If the spec YAML is absent the orchestrator records a ``blocked``
upfront ``run.json`` (status=blocked, reason=spec_not_materialized)
and proceeds; the operator materialises the spec out-of-band (typically
once the upstream dataset manifest for that catalog stanza lands) and
re-invokes the orchestrator. The script never invents specs on the
fly; the dataset/PLM dependencies are tracked separately (LB.1, LR.1
etc.).

Run-record layout
-----------------

Per-cell artefacts live under ``runs/transversal/<shortid>/<cell>/
seed<seed>/``::

    spec.yaml             - the ExperimentSpec that produced this run
    run.json              - run_id, status, axis tuple, shortid, cell,
                            tier, aspect, seed, git_sha, resolved
                            hparams, artefact paths, metrics (when ok)
    model.txt             - LightGBM booster (only when status=ok)
    predictions.parquet   - eval rows + scores (only when status=ok)

The orchestrator writes a placeholder ``run.json`` upfront for every
attempt (status in ``{pending, blocked}``); ``scripts/run.py`` then
overwrites it with the final ``ok`` / ``failed`` record when training
finishes. A crash leaves the most recent state on disk.

CLI surface
-----------

::

    # Dry-run plan: list cells, batches, skip/blocked classification
    python scripts/run_transversal_grid.py --dry-run

    # Real sweep, default batch size 10, seed 42, all 9 tier-aspects
    python scripts/run_transversal_grid.py

    # Restricted: one feature bundle, two seeds, three cells, no champion update
    python scripts/run_transversal_grid.py \
        --features v6+lineage-leakfree --seeds 42,7 \
        --cells nk-bpo,lk-cco,pk-mfo --no-champion-update

    # Limit the first N attempts (useful for smoke tests)
    python scripts/run_transversal_grid.py --limit 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO / "experiments" / "_catalog" / "transversal.yaml"
DEFAULT_SPEC_ROOT = REPO / "experiments" / "_generated" / "transversal"
DEFAULT_RUNS_ROOT = REPO / "runs" / "transversal"
DEFAULT_PYTHON = REPO / ".venv" / "bin" / "python"

DEFAULT_CELLS: tuple[str, ...] = (
    "nk-bpo", "nk-cco", "nk-mfo",
    "lk-bpo", "lk-cco", "lk-mfo",
    "pk-bpo", "pk-cco", "pk-mfo",
)
DEFAULT_SEEDS: tuple[int, ...] = (42,)
DEFAULT_BATCH_SIZE = 10

# Fields propagated from a catalog stanza into the run-record axis
# block (matches the FARM-EXP.3 bootstrap_cis ``axis`` schema).
_AXIS_KEYS: tuple[str, ...] = (
    "plm", "k", "reranker", "features",
    "eval_set", "propagation", "ensemble",
)


# ----------------------------------------------------------- model


@dataclass(frozen=True)
class JobKey:
    """Unique key for one orchestrator attempt."""

    shortid: str
    cell: str
    seed: int

    def relpath(self) -> Path:
        return Path(self.shortid) / self.cell / f"seed{self.seed}"


@dataclass
class JobPlan:
    """Resolved view of a job: spec path, output dir, status sniff."""

    key: JobKey
    stanza: dict[str, Any]
    spec_path: Path
    output_dir: Path
    run_json_path: Path
    spec_exists: bool
    completed: bool


@dataclass
class BatchResult:
    """Outcome of one batch dispatch."""

    attempted: list[JobKey] = field(default_factory=list)
    ok: list[JobKey] = field(default_factory=list)
    failed: list[JobKey] = field(default_factory=list)
    blocked: list[JobKey] = field(default_factory=list)
    skipped: list[JobKey] = field(default_factory=list)


# ----------------------------------------------------------- helpers


def _git_sha(repo: Path = REPO) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_run_id(key: JobKey) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{key.shortid}_{key.cell}_seed{key.seed}"


def _is_complete(run_json_path: Path) -> bool:
    if not run_json_path.exists():
        return False
    try:
        payload = json.loads(run_json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _spec_path_for(
    stanza: dict[str, Any], key: JobKey, *, spec_root: Path,
) -> Path:
    """Resolve the spec yaml path for one ``(stanza, cell, seed)``.

    Default layout: ``<spec_root>/<shortid>/<cell>_seed<seed>.yaml``.
    """
    return spec_root / key.shortid / f"{key.cell}_seed{key.seed}.yaml"


def _axis_block(stanza: dict[str, Any]) -> dict[str, Any]:
    return {k: stanza[k] for k in _AXIS_KEYS if k in stanza}


def _resolve_hparams(spec_path: Path) -> dict[str, Any]:
    """Read ``model.defaults`` + ``training`` from a spec yaml (best-effort)."""
    try:
        payload = yaml.safe_load(spec_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    model = payload.get("model") or {}
    defaults = dict(model.get("defaults") or {})
    training = dict(payload.get("training") or {})
    return {"model": defaults, "training": training}


def _split_cell(cell: str) -> tuple[str | None, str | None]:
    parts = cell.split("-", 1)
    if len(parts) != 2 or not all(parts):
        return (None, None)
    return parts[0], parts[1]


def _relpath_or_abs(path: Path, anchor: Path = REPO) -> str:
    """Return ``path`` relative to ``anchor`` or its absolute form."""
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return str(path)


# ----------------------------------------------------------- planning


def _load_catalog(catalog_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(catalog_path.read_text()) or {}
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(
            f"{catalog_path}: expected top-level 'cells' list"
        )
    return cells


def _filter_stanzas(
    stanzas: Sequence[dict[str, Any]],
    *,
    plms: Sequence[str] | None,
    features: Sequence[str] | None,
    eval_sets: Sequence[str] | None,
    rerankers: Sequence[str] | None,
    shortids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stanza in stanzas:
        if plms and stanza.get("plm") not in plms:
            continue
        if features and stanza.get("features") not in features:
            continue
        if eval_sets and stanza.get("eval_set") not in eval_sets:
            continue
        if rerankers and stanza.get("reranker") not in rerankers:
            continue
        if shortids and stanza.get("shortid") not in shortids:
            continue
        out.append(stanza)
    return out


def _plan_jobs(
    stanzas: Sequence[dict[str, Any]],
    *,
    cells: Sequence[str],
    seeds: Sequence[int],
    spec_root: Path,
    runs_root: Path,
) -> list[JobPlan]:
    plans: list[JobPlan] = []
    for stanza in stanzas:
        shortid = stanza["shortid"]
        for cell in cells:
            for seed in seeds:
                key = JobKey(shortid=shortid, cell=cell, seed=seed)
                spec_path = _spec_path_for(stanza, key, spec_root=spec_root)
                out_dir = runs_root / key.relpath()
                run_json = out_dir / "run.json"
                plans.append(JobPlan(
                    key=key,
                    stanza=stanza,
                    spec_path=spec_path,
                    output_dir=out_dir,
                    run_json_path=run_json,
                    spec_exists=spec_path.exists(),
                    completed=_is_complete(run_json),
                ))
    return plans


def _batches(items: Sequence[JobPlan], size: int) -> Iterable[list[JobPlan]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


# ----------------------------------------------------------- writers


def _write_upfront_run_json(
    plan: JobPlan,
    *,
    run_id: str,
    status: str,
    reason: str | None = None,
) -> None:
    """Persist an upfront run-record placeholder.

    ``status`` is one of ``{pending, blocked}``. ``scripts/run.py``
    overwrites the file with the final record (``ok`` / ``failed``)
    when training finishes. A ``blocked`` record stays in place since
    no training is dispatched for it.
    """
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    tier, aspect = _split_cell(plan.key.cell)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "shortid": plan.key.shortid,
        "cell": plan.key.cell,
        "tier": tier,
        "aspect": aspect,
        "seed": plan.key.seed,
        "axis": _axis_block(plan.stanza),
        "git_sha": _git_sha(),
        "created_at": _now_iso(),
        "spec_path": _relpath_or_abs(plan.spec_path),
        "output_dir": _relpath_or_abs(plan.output_dir),
        "artefact_paths": {
            "model": _relpath_or_abs(plan.output_dir / "model.txt"),
            "predictions": _relpath_or_abs(
                plan.output_dir / "predictions.parquet"
            ),
        },
        "resolved_hparams": _resolve_hparams(plan.spec_path)
        if plan.spec_exists else {},
    }
    if reason:
        payload["reason"] = reason
    plan.run_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


# ----------------------------------------------------------- dispatch


def _python_executable(explicit: Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    if DEFAULT_PYTHON.exists():
        return str(DEFAULT_PYTHON)
    return sys.executable


def _dispatch_one(
    plan: JobPlan,
    *,
    python_exec: str,
    run_py: Path,
) -> tuple[bool, int, str]:
    """Run ``scripts/run.py <spec>`` for one plan; return (ok, code, tail)."""
    if plan.output_dir.exists():
        shutil.rmtree(plan.output_dir)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _build_run_id(plan.key)
    _write_upfront_run_json(plan, run_id=run_id, status="pending")
    proc = subprocess.run(
        [python_exec, str(run_py), str(plan.spec_path)],
        cwd=REPO, capture_output=True, text=True,
    )
    ok = proc.returncode == 0 and _is_complete(plan.run_json_path)
    tail = (proc.stderr or proc.stdout or "").splitlines()[-3:]
    return ok, proc.returncode, "\n    ".join(tail)


def _dispatch_batch(
    batch: Sequence[JobPlan],
    *,
    python_exec: str,
    run_py: Path,
    dry_run: bool,
) -> BatchResult:
    result = BatchResult()
    for plan in batch:
        if plan.completed:
            result.skipped.append(plan.key)
            print(f"  [skip   ] {plan.key.relpath()}  (status=ok)")
            continue
        if not plan.spec_exists:
            result.blocked.append(plan.key)
            if not dry_run:
                run_id = _build_run_id(plan.key)
                _write_upfront_run_json(
                    plan,
                    run_id=run_id,
                    status="blocked",
                    reason="spec_not_materialized",
                )
            print(
                f"  [blocked] {plan.key.relpath()}  "
                f"(missing spec: {_relpath_or_abs(plan.spec_path)})"
            )
            continue
        if dry_run:
            result.attempted.append(plan.key)
            print(f"  [dry    ] {plan.key.relpath()}  spec={plan.spec_path.name}")
            continue
        t0 = time.monotonic()
        ok, exit_code, tail = _dispatch_one(
            plan, python_exec=python_exec, run_py=run_py,
        )
        dur = time.monotonic() - t0
        result.attempted.append(plan.key)
        if ok:
            result.ok.append(plan.key)
            print(f"  [ok     ] {plan.key.relpath()}  ({dur:.0f}s)")
        else:
            result.failed.append(plan.key)
            print(
                f"  [FAIL   ] {plan.key.relpath()}  "
                f"exit={exit_code} dur={dur:.0f}s"
            )
            if tail:
                print(f"      {tail}")
    return result


def _run_champion_update(
    *,
    python_exec: str,
    runs_root: Path,
    update_script: Path,
) -> None:
    """Invoke ``scripts/update_champions.py --apply``; warnings, not errors."""
    proc = subprocess.run(
        [
            python_exec, str(update_script),
            "--runs-dir", str(runs_root),
            "--apply",
        ],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        last = (proc.stdout or "").strip().splitlines()[-1:]
        print(f"  [champion] {''.join(last) or 'updated'}")
    else:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print(
            f"  [champion] WARN exit={proc.returncode}; champion update "
            f"failed (non-fatal):"
        )
        for line in tail:
            print(f"      {line}")


# ----------------------------------------------------------- summary


def _summary(plans: Sequence[JobPlan], results: Sequence[BatchResult]) -> str:
    total = len(plans)
    ok = sum(len(r.ok) for r in results)
    failed = sum(len(r.failed) for r in results)
    blocked = sum(len(r.blocked) for r in results)
    skipped = sum(len(r.skipped) for r in results)
    attempted = sum(len(r.attempted) for r in results)
    dry = attempted - ok - failed
    return (
        f"transversal grid summary: total={total} "
        f"attempted={attempted} ok={ok} failed={failed} "
        f"blocked={blocked} skipped(prev_ok)={skipped} dry={dry}"
    )


# ----------------------------------------------------------- CLI


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_seeds(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return DEFAULT_SEEDS
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--seeds: not an int: {tok!r}"
            ) from exc
    return tuple(out) or DEFAULT_SEEDS


def _add_path_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--catalog", type=Path, default=DEFAULT_CATALOG,
        help="path to the FARM-EXP.2 transversal cell catalog YAML",
    )
    p.add_argument(
        "--spec-root", type=Path, default=DEFAULT_SPEC_ROOT,
        help="root directory holding per-cell ExperimentSpec YAMLs",
    )
    p.add_argument(
        "--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
        help="root for per-cell run artefacts (default runs/transversal)",
    )
    p.add_argument(
        "--python", type=Path, default=None,
        help="python executable for child processes (default: .venv/bin/python)",
    )
    p.add_argument(
        "--run-script", type=Path, default=REPO / "scripts" / "run.py",
        help="path to scripts/run.py",
    )
    p.add_argument(
        "--champion-script", type=Path,
        default=REPO / "scripts" / "update_champions.py",
        help="path to scripts/update_champions.py",
    )


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cells", type=str, default=",".join(DEFAULT_CELLS),
        help="comma-separated tier-aspect cells (subset of nk/lk/pk x bpo/cco/mfo)",
    )
    p.add_argument(
        "--seeds", type=str, default=None,
        help="comma-separated integer seeds (default: 42)",
    )
    p.add_argument("--plms", type=str, default=None, help="filter by PLM names")
    p.add_argument(
        "--features", type=str, default=None,
        help="filter by feature bundles",
    )
    p.add_argument(
        "--eval-sets", type=str, default=None,
        help="filter by eval_set names",
    )
    p.add_argument(
        "--rerankers", type=str, default=None,
        help="filter by reranker spec ids",
    )
    p.add_argument(
        "--shortids", type=str, default=None,
        help="filter by shortid (12-hex)",
    )


def _add_behaviour_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="cells per batch (champion update fires after each batch)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap on planned attempts (applied after planning)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="plan only; no upfront run.json writes, no subprocess",
    )
    p.add_argument(
        "--no-champion-update", action="store_true",
        help="skip scripts/update_champions.py after each batch",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    _add_path_args(p)
    _add_filter_args(p)
    _add_behaviour_args(p)
    return p


# ----------------------------------------------------------- entry


def run(args: argparse.Namespace) -> int:
    stanzas = _load_catalog(args.catalog)
    cells = tuple(_parse_csv(args.cells)) or DEFAULT_CELLS
    seeds = _parse_seeds(args.seeds)
    stanzas = _filter_stanzas(
        stanzas,
        plms=_parse_csv(args.plms) or None,
        features=_parse_csv(args.features) or None,
        eval_sets=_parse_csv(args.eval_sets) or None,
        rerankers=_parse_csv(args.rerankers) or None,
        shortids=_parse_csv(args.shortids) or None,
    )
    plans = _plan_jobs(
        stanzas,
        cells=cells,
        seeds=seeds,
        spec_root=args.spec_root,
        runs_root=args.runs_root,
    )
    if args.limit is not None:
        plans = plans[: args.limit]
    print(
        f"[transversal] catalog={args.catalog.name} "
        f"stanzas_filtered={len(stanzas)} "
        f"cells={len(cells)} seeds={len(seeds)} "
        f"planned_jobs={len(plans)} batch_size={args.batch_size} "
        f"dry_run={args.dry_run}"
    )
    if not plans:
        print("[transversal] no jobs to schedule; exiting.")
        return 0

    python_exec = _python_executable(args.python)
    results: list[BatchResult] = []
    for i, batch in enumerate(_batches(plans, args.batch_size), start=1):
        print(f"[batch {i}] size={len(batch)}")
        result = _dispatch_batch(
            batch,
            python_exec=python_exec,
            run_py=args.run_script,
            dry_run=args.dry_run,
        )
        results.append(result)
        if (
            not args.dry_run
            and not args.no_champion_update
            and (result.ok or result.failed)
        ):
            _run_champion_update(
                python_exec=python_exec,
                runs_root=args.runs_root,
                update_script=args.champion_script,
            )

    print(f"[transversal] {_summary(plans, results)}")
    any_failed = any(r.failed for r in results)
    return 1 if any_failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run(args)


__all__ = [
    "BatchResult",
    "JobKey",
    "JobPlan",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CELLS",
    "DEFAULT_SEEDS",
    "main",
    "run",
]


if __name__ == "__main__":
    sys.exit(main())
