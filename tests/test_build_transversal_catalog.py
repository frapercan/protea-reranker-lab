"""Tests for the FARM-EXP.2 transversal cell catalog generator.

The catalog is the source-of-truth cell list for the F-EXP-RESET
re-benchmark grid. Tests pin:

1. The catalog file is written with the expected stanza shape.
2. Shortids are unique per cell and match the canonical helper.
3. The five pruning rules (R1..R5) are upheld.
4. Cell count lands in the documented constrained band (~120).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from protea_contracts import CANONICAL_AXIS_KEYS, axis_tuple_shortid

import build_study_specs as bss

CELLS = bss.enumerate_cells()


# ---------------------------------------------------------------- count


def test_catalog_cell_count_in_constrained_band() -> None:
    # Axis-map target: ~120 cells (illustrative, see
    # context/experiment-axis-map.md section "What the transversal
    # re-benchmark would cover"). After R1..R5 the constrained grid
    # currently lands at 224 cells (8 PLM x 2 k x balanced
    # reranker/feature/eval mix, with 5 feature bundles including the
    # FARM-EXP.10b ``v6+lineage-leakfree`` value). Band is set wide
    # enough to absorb +/- 40 drift if the axis sweeps are tweaked
    # later, narrow enough to catch a missing rule (which would push
    # the raw cartesian through unfiltered).
    assert 180 <= len(CELLS) <= 280, f"unexpected cell count: {len(CELLS)}"


def test_catalog_writes_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect the catalog output into a tmp path so the test does not
    # touch the committed artefact.
    monkeypatch.setattr(bss, "CATALOG_OUT", tmp_path / "_catalog")
    monkeypatch.setattr(bss, "CATALOG_FILE", tmp_path / "_catalog" / "transversal.yaml")
    n = bss.build_transversal_catalog()
    out = tmp_path / "_catalog" / "transversal.yaml"
    assert out.exists()
    payload = yaml.safe_load(out.read_text())
    assert payload["schema_version"] == "v1"
    assert isinstance(payload["cells"], list)
    assert len(payload["cells"]) == n
    assert 180 <= n <= 280


# ---------------------------------------------------------------- shape


def test_every_stanza_carries_required_keys() -> None:
    required = {
        "shortid",
        "status",
        "plm",
        "k",
        "reranker",
        "features",
        "eval_set",
        "propagation",
        "ensemble",
    }
    for cell in CELLS:
        assert required.issubset(cell.keys()), f"missing keys in {cell}"


def test_status_default_planned() -> None:
    for cell in CELLS:
        assert cell["status"] == "planned"


def test_status_domain_is_documented() -> None:
    # The catalog only emits planned at generation time but downstream
    # writers flip to running/done/superseded; pin the allowed set.
    assert bss.CELL_STATUSES == ("planned", "running", "done", "superseded")


# ---------------------------------------------------------------- shortid


def test_shortid_is_12_hex_lowercase() -> None:
    pattern = re.compile(r"[0-9a-f]{12}")
    for cell in CELLS:
        assert pattern.fullmatch(cell["shortid"]), cell


def test_shortids_are_unique() -> None:
    seen = {c["shortid"] for c in CELLS}
    assert len(seen) == len(CELLS), "shortid collision in transversal catalog"


def test_shortid_matches_canonical_helper() -> None:
    # Recompute via the canonical helper and assert byte-for-byte
    # match. If this drifts, PROTEA's ExperimentRun join breaks.
    for cell in CELLS:
        recomputed = axis_tuple_shortid(bss._axis_payload(cell))
        assert recomputed == cell["shortid"], cell


def test_axis_payload_keys_match_canonical() -> None:
    # The shortid input payload must cover exactly the canonical axis
    # keys (no extras, no omissions). Any drift here is a cross-repo
    # contract violation.
    for cell in CELLS:
        payload = bss._axis_payload(cell)
        assert set(payload.keys()) == set(CANONICAL_AXIS_KEYS)


# ---------------------------------------------------------------- pruning


def test_no_pruned_combo_R1_alignment_weighted_scope() -> None:
    for cell in CELLS:
        if cell["reranker"] == "alignment_weighted":
            assert cell["eval_set"] == "bench-v1-K5-filtered", cell
            assert cell["features"] == "v6", cell


def test_no_pruned_combo_R2_knn_only_lgbm() -> None:
    for cell in CELLS:
        if cell["features"] == "knn-only":
            assert not cell["reranker"].startswith("lgbm."), cell


def test_no_pruned_combo_R3_lineage_eval_needs_lineage_feat() -> None:
    for cell in CELLS:
        if bss._is_lineage_eval(cell["eval_set"]):
            assert "lineage" in cell["features"], cell


def test_no_pruned_combo_R4_geokg_requires_lineage_eval() -> None:
    for cell in CELLS:
        if "geokg" in cell["features"]:
            assert bss._is_lineage_eval(cell["eval_set"]), cell


def test_no_pruned_combo_R5_none_uses_knn_only() -> None:
    for cell in CELLS:
        if cell["reranker"] == "none":
            assert cell["features"] == "knn-only", cell


# ---------------------------------------------------------------- domain


def test_plm_values_in_canonical_set() -> None:
    allowed = set(bss.PLM_SWEEP)
    for cell in CELLS:
        assert cell["plm"] in allowed, cell


def test_k_values_in_canonical_set() -> None:
    allowed = set(bss.K_SWEEP)
    for cell in CELLS:
        assert cell["k"] in allowed, cell


def test_reranker_values_in_canonical_set() -> None:
    allowed = set(bss.RERANKER_SWEEP)
    for cell in CELLS:
        assert cell["reranker"] in allowed, cell


def test_feature_values_in_canonical_set() -> None:
    allowed = set(bss.FEATURE_SWEEP)
    for cell in CELLS:
        assert cell["features"] in allowed, cell


def test_eval_set_values_in_canonical_set() -> None:
    # FARM-EXP.12: the ``lineage`` family expands per-PLM, so the
    # rendered ``eval_set`` is one of:
    #   - ``bench-v1-K5-filtered`` (PLM-blind), or
    #   - ``bench-v1-K{k}-v226-lineage-{plm_short}`` for one of the
    #     nine canonical short keys.
    family_filtered = {"bench-v1-K5-filtered"}
    plm_shorts = set(bss.PLM_SHORT_KEYS.values())
    for cell in CELLS:
        eval_set = cell["eval_set"]
        if eval_set in family_filtered:
            continue
        assert bss._is_lineage_eval(eval_set), cell
        plm_short = eval_set.rsplit("-", 1)[-1]
        assert plm_short in plm_shorts, cell
        # And the suffix must match the cell's own PLM mapping.
        assert plm_short == bss.PLM_SHORT_KEYS[cell["plm"]], cell


def test_propagation_default() -> None:
    for cell in CELLS:
        assert cell["propagation"] == bss.PROPAGATION_DEFAULT


def test_ensemble_default() -> None:
    for cell in CELLS:
        assert cell["ensemble"] == bss.ENSEMBLE_DEFAULT


# ---------------------------------------------------------------- stable


def test_enumerate_cells_is_deterministic() -> None:
    a = bss.enumerate_cells()
    b = bss.enumerate_cells()
    assert [c["shortid"] for c in a] == [c["shortid"] for c in b]


# ---------------------------------------------------------------- FARM-EXP.10b


def test_leakfree_axis_value_is_present() -> None:
    # FARM-EXP.10b adds ``v6+lineage-leakfree`` as a first-class catalog
    # axis value: the FARM-EXP.10 champion config (v6 + lineage minus
    # anc2vec + pca). Pin that it survives pruning on at least one
    # PLM-tagged lineage eval set, so future runner slices can
    # reference it.
    leakfree = [c for c in CELLS if c["features"] == "v6+lineage-leakfree"]
    assert leakfree, "v6+lineage-leakfree absent from catalog"
    assert any(
        bss._is_lineage_eval(c["eval_set"]) for c in leakfree
    ), "v6+lineage-leakfree must be paired with the lineage eval set"


def test_leakfree_bundle_shortid_distinct_from_v6_lineage() -> None:
    # If the leakfree bundle hashed to the same schema_sha as the
    # umbrella v6+lineage bundle, the catalog cells would collide on
    # shortid and silently overlap. Pin distinctness here.
    leakfree_sha = bss._features_to_schema_sha("v6+lineage-leakfree")
    umbrella_sha = bss._features_to_schema_sha("v6+lineage")
    assert leakfree_sha != umbrella_sha
