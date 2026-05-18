#!/usr/bin/env python
"""Generate missing FARM-EXP.9b experiment spec YAML files.

Creates:
- experiments/farm_exp_9/hp_*.yaml for all 27 hparam cells
- experiments/farm_exp_9/standalone_nk-bpo.yaml for bench-v1-K5_nk-bpo_standalone
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO / "experiments" / "farm_exp_9"

LEAKAGE_DROP = [
    "anc2vec_has_emb",
    "anc2vec_neighbor_cos",
    "anc2vec_neighbor_maxcos",
    "anc2vec_query_known_cos",
    "anc2vec_query_known_count",
    "anc2vec_query_known_maxcos",
    "emb_pca_query_0",
    "emb_pca_query_1",
    "emb_pca_query_2",
    "emb_pca_query_3",
    "emb_pca_query_4",
    "emb_pca_query_5",
    "emb_pca_query_6",
    "emb_pca_query_7",
    "emb_pca_query_8",
    "emb_pca_query_9",
    "emb_pca_query_10",
    "emb_pca_query_11",
    "emb_pca_query_12",
    "emb_pca_query_13",
    "emb_pca_query_14",
    "emb_pca_query_15",
]

LR_MAP = {"003": 0.003, "005": 0.05, "01": 0.1}


def _drop_features_yaml(features: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {f}" for f in features)


def gen_hparam_spec(cell_id: str) -> str:
    """Generate YAML for study_v9_hp_L{leaves}_lr{lr}_npr{npr}."""
    m = re.match(r"study_v9_hp_L(\d+)_lr(\d+)_npr(\w+)", cell_id)
    if not m:
        raise ValueError(f"Cannot parse hparam cell_id: {cell_id}")
    leaves_str, lr_raw, npr_raw = m.groups()
    num_leaves = int(leaves_str)
    learning_rate = LR_MAP[lr_raw]
    neg_pos_ratio = None if npr_raw == "none" else int(npr_raw)

    npr_line = (
        f"    neg_pos_ratio: {neg_pos_ratio}\n" if neg_pos_ratio is not None else ""
    )
    drop_yaml = _drop_features_yaml(LEAKAGE_DROP)

    spec_name = f"farm_exp_9_hp_L{leaves_str}_lr{lr_raw}_npr{npr_raw}"
    output_dir = f"runs/transversal/{spec_name}"

    desc = (
        f"FARM-EXP.9 hparam sweep point: num_leaves={num_leaves}, "
        f"lr={learning_rate}, neg_pos_ratio={neg_pos_ratio}. "
        f"Cell=nk-bpo, seed=42. Re-runs on bench-v1-K5-filtered. "
        f"NOT comparable to pre-leakage bench-v1-K5 runs."
    )

    return f"""schema_version: v1
name: {spec_name}
description: "{desc}"
dataset:
  manifest: datasets/bench-v1-K5-filtered/manifest.json
model:
  kind: lgbm_reranker
  defaults:
    objective: lambdarank
    num_boost_round: 5000
    early_stopping_rounds: 50
    learning_rate: {learning_rate}
    num_leaves: {num_leaves}
    min_data_in_leaf: 100
{npr_line}    drop_features:
{drop_yaml}
training:
  cell: nk-bpo
  val_strategy: temporal
  val_holdout_snapshot: v215-v220
  val_fraction: 0.2
  seed: 42
  propagate_labels: false
sweep:
  backend: none
output_dir: {output_dir}
tags:
- farm_exp_9
- study_v9_hparam_rerun
- bench-v1-K5-filtered
- nk-bpo
- seed42
- L{leaves_str}_lr{lr_raw}_npr{npr_raw}
keep_staging: false
"""


def gen_standalone_spec() -> str:
    """Generate YAML for bench-v1-K5_nk-bpo_standalone."""
    drop_yaml = _drop_features_yaml(LEAKAGE_DROP)
    spec_name = "farm_exp_9_standalone_nk-bpo_seed42"
    return f"""schema_version: v1
name: {spec_name}
description: "FARM-EXP.9 standalone re-run. Cell=nk-bpo, seed=42 on bench-v1-K5-filtered. Earliest bench-v1-K5 nk-bpo run. Overlaps with study_v9_rep_nk-bpo_seed42 logically. NOT comparable to pre-leakage runs."
dataset:
  manifest: datasets/bench-v1-K5-filtered/manifest.json
model:
  kind: lgbm_reranker
  defaults:
    objective: lambdarank
    num_boost_round: 5000
    early_stopping_rounds: 50
    learning_rate: 0.05
    num_leaves: 63
    min_data_in_leaf: 100
    drop_features:
{drop_yaml}
training:
  cell: nk-bpo
  val_strategy: temporal
  val_holdout_snapshot: v215-v220
  val_fraction: 0.2
  seed: 42
  propagate_labels: false
sweep:
  backend: none
output_dir: runs/transversal/{spec_name}
tags:
- farm_exp_9
- standalone_rerun
- bench-v1-K5-filtered
- nk-bpo
- seed42
keep_staging: false
"""


HPARAM_CELL_IDS = [
    "study_v9_hp_L127_lr003_npr10",
    "study_v9_hp_L127_lr003_npr5",
    "study_v9_hp_L127_lr003_nprnone",
    "study_v9_hp_L127_lr005_npr10",
    "study_v9_hp_L127_lr005_npr5",
    "study_v9_hp_L127_lr005_nprnone",
    "study_v9_hp_L127_lr01_npr10",
    "study_v9_hp_L127_lr01_npr5",
    "study_v9_hp_L127_lr01_nprnone",
    "study_v9_hp_L31_lr003_npr10",
    "study_v9_hp_L31_lr003_npr5",
    "study_v9_hp_L31_lr003_nprnone",
    "study_v9_hp_L31_lr005_npr10",
    "study_v9_hp_L31_lr005_npr5",
    "study_v9_hp_L31_lr005_nprnone",
    "study_v9_hp_L31_lr01_npr10",
    "study_v9_hp_L31_lr01_npr5",
    "study_v9_hp_L31_lr01_nprnone",
    "study_v9_hp_L63_lr003_npr10",
    "study_v9_hp_L63_lr003_npr5",
    "study_v9_hp_L63_lr003_nprnone",
    "study_v9_hp_L63_lr005_npr10",
    "study_v9_hp_L63_lr005_npr5",
    "study_v9_hp_L63_lr005_nprnone",
    "study_v9_hp_L63_lr01_npr10",
    "study_v9_hp_L63_lr01_npr5",
    "study_v9_hp_L63_lr01_nprnone",
]


def main() -> int:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    # Generate hparam specs
    for cell_id in HPARAM_CELL_IDS:
        m = re.match(r"study_v9_hp_L(\d+)_lr(\d+)_npr(\w+)", cell_id)
        if not m:
            continue
        leaves_str, lr_raw, npr_raw = m.groups()
        filename = f"hp_L{leaves_str}_lr{lr_raw}_npr{npr_raw}.yaml"
        path = SPEC_DIR / filename
        if path.exists():
            print(f"[skip] {filename} already exists")
            continue
        path.write_text(gen_hparam_spec(cell_id))
        print(f"[wrote] {filename}")

    # Generate standalone spec
    standalone_path = SPEC_DIR / "standalone_nk-bpo.yaml"
    if standalone_path.exists():
        print("[skip] standalone_nk-bpo.yaml already exists")
    else:
        standalone_path.write_text(gen_standalone_spec())
        print("[wrote] standalone_nk-bpo.yaml")

    print(f"\nTotal generated: {len(HPARAM_CELL_IDS)} hparam + 1 standalone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
