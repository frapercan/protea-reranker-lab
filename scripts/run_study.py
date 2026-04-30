#!/usr/bin/env python
"""Sequential, resumable orchestrator for the v9 study phases.

Each spec produces its own ``output_dir/run.json``; this script:
  1. iterates the YAMLs of a phase in deterministic name order;
  2. skips a spec if its ``run.json`` already says ``status=ok``;
  3. otherwise runs ``scripts/run.py <spec>`` as a subprocess and waits;
  4. appends a row to ``runs/study_v9/<phase>/results.csv`` per success/failure.

Per phase the orchestrator is a dumb sequential loop — total wall-clock is the
sum of per-spec durations. There is no parallelism (RAM budget would force
care otherwise; a future ``--workers N`` toggle is a separate change).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_ROOT = REPO / "experiments" / "_generated" / "study_v9"
RESULTS_ROOT = REPO / "runs" / "study_v9"
PYTHON = REPO / ".venv" / "bin" / "python"


PHASE_DIRS = {
    "f1": "f1_replication",
    "f2": "f2_ablation",
    "f4": "f4_hparam",
}
RESULTS_SUBDIR = {
    "f1": "replication",
    "f2": "ablation",
    "f4": "hparam",
}


def _is_complete(run_json: Path) -> bool:
    if not run_json.exists():
        return False
    try:
        return json.loads(run_json.read_text()).get("status") == "ok"
    except Exception:
        return False


def _read_metrics(run_json: Path) -> dict[str, object]:
    try:
        report = json.loads(run_json.read_text())
    except Exception:
        return {}
    m = report.get("metrics", {}) or {}
    s = report.get("split", {}) or {}
    return {
        "fmax": m.get("test_fmax"),
        "best_iteration": m.get("best_iteration"),
        "n_train": s.get("n_train"),
        "n_val": s.get("n_val"),
        "n_eval": s.get("n_eval"),
        "duration_s": report.get("duration_s"),
        "status": report.get("status"),
    }


def _output_dir_of(spec_yaml: Path) -> Path:
    import yaml
    payload = yaml.safe_load(spec_yaml.read_text())
    return REPO / payload["output_dir"]


def _append_csv(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_phase(phase: str, *, dry_run: bool = False, only: list[str] | None = None) -> int:
    phase_dir = SPEC_ROOT / PHASE_DIRS[phase]
    if not phase_dir.exists():
        print(f"[err] missing spec dir {phase_dir}; run scripts/build_study_specs.py first")
        return 2
    specs = sorted(phase_dir.glob("*.yaml"))
    if only:
        specs = [s for s in specs if s.stem in only]
    csv_path = RESULTS_ROOT / RESULTS_SUBDIR[phase] / "results.csv"

    print(f"[study_v9] phase={phase}  specs={len(specs)}  results→{csv_path}")
    failed: list[str] = []
    for i, spec in enumerate(specs, 1):
        out_dir = _output_dir_of(spec)
        run_json = out_dir / "run.json"
        if _is_complete(run_json):
            metrics = _read_metrics(run_json)
            print(f"[skip {i}/{len(specs)}] {spec.stem}  fmax={metrics.get('fmax')}")
            continue
        if dry_run:
            print(f"[dry  {i}/{len(specs)}] {spec.stem}")
            continue

        # Clean partial run dir so we get a clean slate
        if out_dir.exists():
            shutil.rmtree(out_dir)

        t0 = time.monotonic()
        print(f"[run  {i}/{len(specs)}] {spec.stem}  …", flush=True)
        proc = subprocess.run(
            [str(PYTHON), str(REPO / "scripts" / "run.py"), str(spec)],
            cwd=REPO, capture_output=True, text=True,
        )
        dur = time.monotonic() - t0
        ok = proc.returncode == 0 and _is_complete(run_json)
        metrics = _read_metrics(run_json)
        row = {
            "spec": spec.stem,
            "phase": phase,
            "status": "ok" if ok else "failed",
            "exit_code": proc.returncode,
            "fmax": metrics.get("fmax"),
            "best_iteration": metrics.get("best_iteration"),
            "n_train": metrics.get("n_train"),
            "n_val": metrics.get("n_val"),
            "n_eval": metrics.get("n_eval"),
            "duration_s": round(dur, 2),
            "output_dir": str(out_dir.relative_to(REPO)),
        }
        _append_csv(csv_path, row)
        if ok:
            print(f"[done {i}/{len(specs)}] {spec.stem}  fmax={metrics.get('fmax')}  "
                  f"dur={dur:.0f}s  best_iter={metrics.get('best_iteration')}")
        else:
            failed.append(spec.stem)
            tail = (proc.stderr or proc.stdout or "").splitlines()[-5:]
            print(f"[FAIL {i}/{len(specs)}] {spec.stem}  exit={proc.returncode}  "
                  f"dur={dur:.0f}s")
            for line in tail:
                print(f"    {line}")

    if failed:
        print(f"[study_v9] phase={phase} FAILED: {failed}")
        return 1
    print(f"[study_v9] phase={phase} COMPLETE")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=sorted(PHASE_DIRS), help="which phase to run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="+", help="run only specs whose stem is in this list")
    args = p.parse_args(argv)
    return run_phase(args.phase, dry_run=args.dry_run, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
