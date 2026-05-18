#!/usr/bin/env python
"""Generate ExperimentSpec YAMLs for the v9 study and the F-EXP-RESET catalog.

The v9 path emits the historical 9-cell study (f1 replication, f2
ablation, f4 hparam) into ``experiments/_generated/study_v9/``.

The transversal catalog path (FARM-EXP.2) emits a single
``experiments/_catalog/transversal.yaml`` listing every materialised
cell in the F-EXP-RESET re-benchmark grid (one YAML stanza per cell).

Output tree (relative to repo root):

    experiments/_generated/study_v9/
    f1_replication/<cell>_seed<N>.yaml          (27 specs: 9 cells x 3 seeds)
    f2_ablation/<cell>_drop_<family>.yaml       (39 specs: 3 cells x 13 families)
    f4_hparam/nkbpo_L<leaves>_lr<lr>_npr<npr>.yaml  (27 specs: 1 cell x 3 cubed grid)

    experiments/_catalog/transversal.yaml         (~112 axis-tuple cells)

All v9 specs target ``datasets/bench-v1-K5/manifest.json`` directly via
the ``dataset.manifest`` form (no derived dataset materialization).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from protea_contracts import (
    CANONICAL_AXIS_KEYS,
    axis_tuple_shortid,
)

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


# ---------------------------------------------------------------------------
# FARM-EXP.2 - transversal cell catalog
# ---------------------------------------------------------------------------
#
# One YAML stanza per axis-tuple cell, written to
# ``experiments/_catalog/transversal.yaml``. The catalog is the
# source-of-truth list consumed by FARM-EXP.3+ runner slices. Pruning
# rules below mirror the axis-map section "What the transversal
# re-benchmark would cover" plus two principled extensions documented in
# results/executor-1778953289-10fa/plan.md.

CATALOG_OUT = Path(__file__).resolve().parents[1] / "experiments" / "_catalog"
CATALOG_FILE = CATALOG_OUT / "transversal.yaml"

#: 8-PLM list verified against PROTEA/EXPERIMENTAL_DESIGN.md section 4.
PLM_SWEEP: tuple[str, ...] = (
    "esmc_300m",
    "esmc_600m",
    "esm2_650m",
    "esm2_3b",
    "ankh_base",
    "ankh_large",
    "prot_t5_xl",
    "prostt5_xl",
)

K_SWEEP: tuple[int, ...] = (5, 10)

RERANKER_SWEEP: tuple[str, ...] = (
    "none",
    "alignment_weighted",
    "lgbm.per_tier_3",
    "lgbm.per_cell_9",
)

FEATURE_SWEEP: tuple[str, ...] = (
    "knn-only",
    "v6",
    "v6+lineage",
    "v6+lineage-leakfree",
    "v6+lineage+geokg",
)
#: ``v6+lineage-leakfree`` is the FARM-EXP.10 leakage-fixed champion
#: bundle: v6 + lineage families minus the ``anc2vec_neighbor``,
#: ``anc2vec_query`` and ``emb_pca`` families (anc2vec + PCA caused
#: a known-label shortcut on v226-lineage; see memory
#: ``project_lb2_leakage_fixed_champion``). The catalog tracks it by
#: name; the runner slice (FARM-EXP.5+) resolves the name to the actual
#: family list / drop list when emitting run records.

EVAL_SWEEP: tuple[str, ...] = (
    "bench-v1-K5-filtered",
    "bench-v1-K5-v226-lineage",
)

PROPAGATION_DEFAULT = "tpr_pred"
ENSEMBLE_DEFAULT = "none"

#: Allowed stanza status values, mirroring run lifecycle.
CELL_STATUSES: tuple[str, ...] = ("planned", "running", "done", "superseded")


def _features_to_schema_sha(features: str) -> str:
    """Map a feature-bundle name to a stable 12-hex digest.

    The catalog rides on bundle names (``knn-only``, ``v6``, ...) so the
    axis tuple stays human-readable. The shortid input needs a stable
    feature-schema digest (one column of the canonical axis tuple). We
    derive it deterministically from the bundle name; the runner slice
    will later replace this with the live ``compute_feature_schema_sha``
    digest once the bundle-to-column mapping is wired.
    """
    blob = f"feature-bundle:{features}".encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _eval_set_manifest_sha(eval_set: str) -> str:
    """Resolve the eval-set manifest sha or fall back to a placeholder.

    ``bench-v1-K5-v226-lineage`` may not be materialised yet (depends on
    lab-runner LB.1); for catalog generation we accept the placeholder
    and let FARM-EXP.3 backfill once the manifest lands.
    """
    manifest = Path(__file__).resolve().parents[1] / "datasets" / eval_set / "manifest.json"
    if manifest.exists():
        return hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    return f"manifest-tbd-{eval_set}"


def _axis_payload(stanza: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical axis-tuple payload from a catalog stanza.

    Keys match :data:`protea_contracts.CANONICAL_AXIS_KEYS` so the
    shortid is shared byte-for-byte with PROTEA's ``ExperimentRun``
    ``axis_tuple_shortid`` column.
    """
    return {
        "plm": stanza["plm"],
        "k": stanza["k"],
        "reranker_spec_id": stanza["reranker"],
        "feature_schema_sha": _features_to_schema_sha(stanza["features"]),
        "eval_set_name": stanza["eval_set"],
        "eval_set_manifest_sha": _eval_set_manifest_sha(stanza["eval_set"]),
        "propagation": stanza["propagation"],
        "ensemble_spec": stanza["ensemble"],
    }


def _keep_cell(cell: dict[str, Any]) -> bool:
    """Apply pruning rules R1..R5.

    R1: ``alignment_weighted`` only on (eval=bench-v1-K5-filtered,
        features=v6). Dominated by lgbm.* elsewhere.
    R2: drop (features=knn-only, reranker=lgbm.*). Degenerate.
    R3: drop (eval=bench-v1-K5-v226-lineage) with features lacking
        ``lineage``. The lineage dataset exists to test the lineage
        feature family.
    R4: drop (features contains ``geokg``) with eval != lineage.
        geokg requires the lineage eval set per LR.2.
    R5: drop (reranker=none) with features != knn-only. The KNN-only
        baseline by definition runs on the knn-only feature bundle.
    """
    if cell["reranker"] == "alignment_weighted":
        if cell["eval_set"] != "bench-v1-K5-filtered" or cell["features"] != "v6":
            return False
    if cell["features"] == "knn-only" and cell["reranker"].startswith("lgbm."):
        return False
    if cell["eval_set"] == "bench-v1-K5-v226-lineage" and "lineage" not in cell["features"]:
        return False
    if "geokg" in cell["features"] and cell["eval_set"] != "bench-v1-K5-v226-lineage":
        return False
    if cell["reranker"] == "none" and cell["features"] != "knn-only":
        return False
    return True


def enumerate_cells() -> list[dict[str, Any]]:
    """Return the constrained list of catalog stanzas with shortids.

    Pure function: depends only on module constants + the canonical
    shortid helper. Stable under repeated calls (sorted output).
    """
    raw: list[dict[str, Any]] = []
    for plm, k, rr, feat, eval_set in product(
        PLM_SWEEP, K_SWEEP, RERANKER_SWEEP, FEATURE_SWEEP, EVAL_SWEEP
    ):
        raw.append(
            {
                "plm": plm,
                "k": k,
                "reranker": rr,
                "features": feat,
                "eval_set": eval_set,
                "propagation": PROPAGATION_DEFAULT,
                "ensemble": ENSEMBLE_DEFAULT,
            }
        )
    kept = [c for c in raw if _keep_cell(c)]
    for cell in kept:
        cell["shortid"] = axis_tuple_shortid(_axis_payload(cell))
        cell["status"] = "planned"
    # Sort by shortid for deterministic ordering across regenerations.
    kept.sort(key=lambda c: c["shortid"])
    return kept


def _stanza_for_yaml(cell: dict[str, Any]) -> dict[str, Any]:
    """Order keys for human-readable YAML output."""
    return {
        "shortid": cell["shortid"],
        "status": cell["status"],
        "plm": cell["plm"],
        "k": cell["k"],
        "reranker": cell["reranker"],
        "features": cell["features"],
        "eval_set": cell["eval_set"],
        "propagation": cell["propagation"],
        "ensemble": cell["ensemble"],
    }


CATALOG_HEADER = """\
# Transversal cell catalog (FARM-EXP.2)
#
# Generated by scripts/build_study_specs.py --transversal-catalog. Do
# not edit by hand; regenerate with the above command.
#
# Each stanza is one axis-tuple cell. The shortid is the canonical
# protea_contracts.axis_tuple_shortid digest of the axis payload
# (matches PROTEA's ExperimentRun.axis_tuple_shortid column).
#
# Pruning rules applied (see axis-map section "What the transversal
# re-benchmark would cover"):
#   R1: alignment_weighted only on (eval=bench-v1-K5-filtered, features=v6)
#   R2: drop (features=knn-only, reranker=lgbm.*)
#   R3: drop (eval=bench-v1-K5-v226-lineage) with features lacking lineage
#   R4: drop (features contains geokg) with eval != lineage
#   R5: drop (reranker=none) with features != knn-only
"""


def build_transversal_catalog() -> int:
    """Write the transversal catalog and return the cell count."""
    cells = enumerate_cells()
    CATALOG_OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "cells": [_stanza_for_yaml(c) for c in cells],
    }
    body = yaml.safe_dump(payload, sort_keys=False)
    CATALOG_FILE.write_text(CATALOG_HEADER + body)
    return len(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate study YAMLs.")
    parser.add_argument(
        "--transversal-catalog",
        action="store_true",
        help="Emit experiments/_catalog/transversal.yaml (FARM-EXP.2).",
    )
    args = parser.parse_args(argv)

    if args.transversal_catalog:
        n = build_transversal_catalog()
        print(f"[transversal] wrote {n} cells -> {CATALOG_FILE}")
        return 0

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    n_f1 = build_f1()
    n_f2 = build_f2()
    n_f4 = build_f4()
    print(f"[study_v9] f1 replication: {n_f1} specs -> {OUT_ROOT/'f1_replication'}")
    print(f"[study_v9] f2 ablation:    {n_f2} specs -> {OUT_ROOT/'f2_ablation'}")
    print(f"[study_v9] f4 hparam:      {n_f4} specs -> {OUT_ROOT/'f4_hparam'}")
    print(f"[study_v9] total: {n_f1 + n_f2 + n_f4} specs")
    return 0


__all__ = [
    "CANONICAL_AXIS_KEYS",
    "CATALOG_FILE",
    "CELL_STATUSES",
    "EVAL_SWEEP",
    "FEATURE_SWEEP",
    "K_SWEEP",
    "PLM_SWEEP",
    "RERANKER_SWEEP",
    "build_transversal_catalog",
    "enumerate_cells",
]


if __name__ == "__main__":
    import sys
    sys.exit(main())
