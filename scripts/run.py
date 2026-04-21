#!/usr/bin/env python
"""Run one ExperimentSpec end-to-end.

Usage:
    # Manual single run
    python scripts/run.py experiments/example.yaml

    # Manual with hparam override
    python scripts/run.py experiments/example.yaml \\
        --hparam learning_rate=0.1 --hparam num_leaves=31

    # Launched by wandb agent — sweep-injected flags ride on wandb.config
    wandb agent <sweep_id>

Writes artefacts to ``spec.output_dir`` (or ``runs/<run_id>``) — always leaves
``spec.yaml`` + ``run.json`` behind, even on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from protea_reranker_lab import ExperimentSpec
from protea_reranker_lab.runner import run_experiment


def _parse_hparam(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--hparam requires key=value, got {raw!r}")
    k, v = raw.split("=", 1)
    # JSON-decode if it parses (ints, floats, bools, null, lists) — else string.
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", help="path to ExperimentSpec YAML")
    p.add_argument("--datasets-root", default="datasets",
                   help="root dir where derived datasets are materialized")
    p.add_argument("--hparam", action="append", type=_parse_hparam, default=[],
                   help="override a model.defaults or training field (repeatable)")
    # parse_known_args so wandb-agent-injected --key=value flags don't crash us;
    # those come through via wandb.config after wandb.init() inside the runner.
    args, _unknown = p.parse_known_args(argv)

    overrides: dict[str, Any] = dict(args.hparam)

    spec = ExperimentSpec.from_yaml(args.spec)
    print(f"[run] {spec.name}  spec_hash={spec.hash()}  tags={spec.tags or '—'}")
    if overrides:
        print(f"[run] cli overrides: {overrides}")
    if spec.sweep.backend == "wandb":
        print(f"[run] wandb backend project={spec.sweep.project}")

    report = run_experiment(
        spec,
        datasets_root=args.datasets_root,
        hparam_overrides=overrides or None,
    )

    m = report.get("metrics", {})
    s = report.get("split", {})
    ds = report.get("dataset", {})
    h = report.get("resolved_hparams", {})
    feat = report.get("features", {})
    w = report.get("wandb") or {}

    print(f"[done] status={report['status']}  run_id={report['run_id']}  "
          f"duration={report['duration_s']}s")
    print(f"       test_fmax={m.get('test_fmax', 0):.4f}  "
          f"best_iter={m.get('best_iteration', 0)}  "
          f"train={s.get('n_train', 0):,} val={s.get('n_val', 0):,} eval={s.get('n_eval', 0):,}")
    print(f"       dataset={ds.get('name')}  schema_sha={ds.get('schema_sha')}  "
          f"families={feat.get('families_enabled') or 'ALL'}  "
          f"features={feat.get('feature_count')}")
    print(f"       hparams: obj={h.get('objective')} lr={h.get('learning_rate')} "
          f"leaves={h.get('num_leaves')} min_leaf={h.get('min_data_in_leaf')} "
          f"neg_pos={h.get('neg_pos_ratio')} seed={h.get('seed')}")
    if w:
        print(f"       wandb: {w.get('url')}")
    print(f"[artefacts] {report['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
