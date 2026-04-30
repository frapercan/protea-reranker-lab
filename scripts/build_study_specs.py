#!/usr/bin/env python
"""Generate ExperimentSpec YAMLs for the v9 study (f1 replication, f2 ablation, f4 hparam).

The plan is congealed in memory `project_study_v9_plan.md`. This script is the
single source of truth for *which* YAMLs we run. It does NOT execute anything;
``scripts/run_study.py`` consumes the generated YAMLs.

Output tree (relative to repo root):

    experiments/_generated/study_v9/
    ├── f1_replication/<cell>_seed<N>.yaml          (27 specs: 9 cells × 3 seeds)
    ├── f2_ablation/<cell>_drop_<family>.yaml       (39 specs: 3 cells × 13 families)
    └── f4_hparam/nkbpo_L<leaves>_lr<lr>_npr<npr>.yaml  (27 specs: 1 cell × 3³ grid)

All specs target ``datasets/bench-v1-K5/manifest.json`` directly via the
``dataset.manifest`` form (no derived dataset materialization).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "experiments" / "_generated" / "study_v9"
RUNS_ROOT = "runs/study_v9"  # spec.output_dir prefix (relative path stays portable)
DATASET_MANIFEST = "datasets/bench-v1-K5/manifest.json"

CELLS_FULL = [
    "nk-bpo", "nk-mfo", "nk-cco",
    "lk-bpo", "lk-mfo", "lk-cco",
    "pk-bpo", "pk-mfo", "pk-cco",
]
CELLS_REPRESENTATIVE = ["nk-bpo", "lk-cco", "pk-mfo"]
SEEDS_F1 = [42, 7, 137]

FEATURE_FAMILIES = [
    "knn_distance", "knn_vote",
    "alignment_nw", "alignment_sw",
    "length",
    "taxonomy_pair", "taxonomy_voters",
    "go_context",
    "anc2vec_neighbor", "anc2vec_query",
    "emb_pca",
    "annotation_meta",
    "knn",  # superset of knn_distance + knn_vote (kept for completeness; collinear with the two)
]

# Hyperparam grid for F4 (nk-bpo only)
F4_NUM_LEAVES = [31, 63, 127]
F4_LR = [0.03, 0.05, 0.10]
F4_NEG_POS = [None, 10.0, 5.0]


def _write_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _base_spec(name: str, cell: str, *, output_dir: str, tags: list[str],
               num_boost_round: int, drop_features: list[str] | None = None,
               num_leaves: int = 63, learning_rate: float = 0.05,
               min_data_in_leaf: int = 100, neg_pos_ratio: float | None = None,
               seed: int = 42) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "objective": "lambdarank",
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": 50,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
    }
    if drop_features:
        defaults["drop_features"] = list(drop_features)
    if neg_pos_ratio is not None:
        defaults["neg_pos_ratio"] = neg_pos_ratio
    return {
        "name": name,
        "dataset": {"manifest": DATASET_MANIFEST},
        "model": {"kind": "lgbm_reranker", "defaults": defaults},
        "training": {
            "cell": cell,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": seed,
        },
        "sweep": {"backend": "none"},
        "output_dir": output_dir,
        "tags": tags,
    }


def build_f1() -> int:
    out = OUT_ROOT / "f1_replication"
    n = 0
    for cell in CELLS_FULL:
        for seed in SEEDS_F1:
            name = f"v9_f1_{cell}_seed{seed}"
            spec = _base_spec(
                name=name, cell=cell,
                output_dir=f"{RUNS_ROOT}/replication/{cell}_seed{seed}",
                tags=["study_v9", "f1_replication", cell, f"seed{seed}"],
                num_boost_round=5000,
                seed=seed,
            )
            _write_spec(out / f"{cell}_seed{seed}.yaml", spec)
            n += 1
    return n


def build_f2() -> int:
    out = OUT_ROOT / "f2_ablation"
    n = 0
    # FEATURE_FAMILIES is a dict mapping family -> column list; we drop those columns.
    from protea_reranker_lab.reranker import FEATURE_FAMILIES as FAM_MAP
    for cell in CELLS_REPRESENTATIVE:
        for family in FEATURE_FAMILIES:
            cols_to_drop = list(FAM_MAP.get(family, []))
            if not cols_to_drop:
                continue
            name = f"v9_f2_{cell}_drop_{family}"
            spec = _base_spec(
                name=name, cell=cell,
                output_dir=f"{RUNS_ROOT}/ablation/{cell}_drop_{family}",
                tags=["study_v9", "f2_ablation", cell, f"drop_{family}"],
                num_boost_round=2000,
                drop_features=cols_to_drop,
                seed=42,
            )
            _write_spec(out / f"{cell}_drop_{family}.yaml", spec)
            n += 1
    return n


def build_f4() -> int:
    out = OUT_ROOT / "f4_hparam"
    n = 0
    cell = "nk-bpo"
    for leaves in F4_NUM_LEAVES:
        for lr in F4_LR:
            for npr in F4_NEG_POS:
                npr_tag = "none" if npr is None else f"{npr:g}"
                lr_tag = f"{lr:g}".replace(".", "")
                name = f"v9_f4_nkbpo_L{leaves}_lr{lr_tag}_npr{npr_tag}"
                spec = _base_spec(
                    name=name, cell=cell,
                    output_dir=f"{RUNS_ROOT}/hparam/L{leaves}_lr{lr_tag}_npr{npr_tag}",
                    tags=["study_v9", "f4_hparam", cell,
                          f"leaves{leaves}", f"lr{lr_tag}", f"npr{npr_tag}"],
                    num_boost_round=2000,
                    num_leaves=leaves, learning_rate=lr,
                    neg_pos_ratio=npr,
                    seed=42,
                )
                _write_spec(out / f"L{leaves}_lr{lr_tag}_npr{npr_tag}.yaml", spec)
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    n_f1 = build_f1()
    n_f2 = build_f2()
    n_f4 = build_f4()
    print(f"[study_v9] f1 replication: {n_f1} specs → {OUT_ROOT/'f1_replication'}")
    print(f"[study_v9] f2 ablation:    {n_f2} specs → {OUT_ROOT/'f2_ablation'}")
    print(f"[study_v9] f4 hparam:      {n_f4} specs → {OUT_ROOT/'f4_hparam'}")
    print(f"[study_v9] total: {n_f1 + n_f2 + n_f4} specs")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
