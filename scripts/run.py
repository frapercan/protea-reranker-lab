#!/usr/bin/env python
"""Run one ExperimentSpec end-to-end.

Usage:
    python scripts/run.py experiments/example.yaml

Writes artefacts to ``spec.output_dir`` (or ``runs/<run_id>``) — always leaves
``spec.yaml`` + ``run.json`` behind, even on failure.
"""

from __future__ import annotations

import argparse
import sys

from protea_reranker_lab import ExperimentSpec
from protea_reranker_lab.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", help="path to ExperimentSpec YAML")
    p.add_argument("--datasets-root", default="datasets",
                   help="root dir where derived datasets are materialized")
    args = p.parse_args(argv)

    spec = ExperimentSpec.from_yaml(args.spec)
    print(f"[run] {spec.name}  spec_hash={spec.hash()}  tags={spec.tags or '—'}")

    report = run_experiment(spec, datasets_root=args.datasets_root)

    m = report.get("metrics", {})
    s = report.get("split", {})
    ds = report.get("dataset", {})
    h = report.get("resolved_hparams", {})
    feat = report.get("features", {})

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
    print(f"[artefacts] {report['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
