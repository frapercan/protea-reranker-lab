"""MR-2 combiner mode: shallow per-category meta-learner over the score vector.

Covers the additive combiner training mode wired into
:mod:`protea_reranker_lab.train` / :mod:`protea_reranker_lab.runner`:

- The score-vector column contract (``combiner`` module) stays in sync with
  PROTEA's EvidenceScorer adapters: NK drops the two priors, LK / PK keep them.
- ``TrainConfig.feature_override`` restricts ``selected_features()`` to exactly
  the supplied columns, bypassing the contracts families, while still honouring
  ``drop_features``.
- ``train._build_spec`` honours ``--combiner`` / ``--combiner-columns`` and
  shrinks ``num_leaves`` only when the operator left it at the monolith default.
- End to end on a synthetic parquet: the combiner trains per cell, emits the
  registerable artifact triple (``model.txt`` / ``spec.yaml`` / ``run.json``),
  the booster's baked ``feature_name`` is exactly the score vector, and the
  run.json records the combiner provenance. The monolith path is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from protea_reranker_lab.combiner import (
    DEFAULT_COMBINER_COLUMNS,
    SCORE_VECTOR_BY_SCORER,
    combiner_columns_for_category,
    resolve_combiner_columns,
)
from protea_reranker_lab.reranker import TrainConfig
from protea_reranker_lab.train import (
    _COMBINER_NEG_POS_RATIO_DEFAULT,
    _COMBINER_NUM_LEAVES_DEFAULT,
    _MONOLITH_NUM_LEAVES_DEFAULT,
    _build_spec,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Score-vector column contract (mirror of PROTEA scorers.py)
# ---------------------------------------------------------------------------


#: The three base-evidence columns (sequence / taxonomy / label-embedding) that
#: apply to every category and lead the vector. The IA-harmful go_term_frequency
#: and export-only interpro_score columns were dropped (MR-2.5).
_BASE_EVIDENCE_COLUMNS = [
    "alignment_score_sw",
    "taxonomic_distance",
    "anc2vec_neighbor_maxcos",
]
#: The knn + classifier columns: also all-category, after the base evidence.
_KNN_CLF_COLUMNS = ["distance", "neighbor_vote_fraction", "classifier_score"]
#: The two prior columns dropped for NK.
_PRIOR_COLUMNS = ["self_prior_score", "association_total", "association_cross"]


def test_default_columns_are_base_evidence_then_knn_clf_then_priors() -> None:
    """DEFAULT_COMBINER_COLUMNS = base evidence, then knn/classifier, then priors."""
    assert DEFAULT_COMBINER_COLUMNS == [
        "alignment_score_sw",
        "taxonomic_distance",
        "anc2vec_neighbor_maxcos",
        "distance",
        "neighbor_vote_fraction",
        "classifier_score",
        "self_prior_score",
        "association_total",
        "association_cross",
    ]


def test_ia_harmful_columns_are_not_in_the_score_vector() -> None:
    """go_term_frequency + interpro_score were dropped (MR-2.5).

    go_term_frequency biases toward FREQUENT low-IA terms (the IA-weighted
    f_micro_w rewards RARE informative terms); interpro_score is export-only
    (0 percent importance, not computable at PROTEA predict time). Neither may
    re-enter the default vector for any category.
    """
    for harmful in ("go_term_frequency", "interpro_score"):
        assert harmful not in DEFAULT_COMBINER_COLUMNS
        for cat in ("nk", "lk", "pk"):
            assert harmful not in combiner_columns_for_category(cat)
    for scorer in ("term_frequency", "interpro"):
        assert scorer not in SCORE_VECTOR_BY_SCORER


def test_scorer_map_keys_match_protea_registry_order() -> None:
    """The scorer names + order mirror PROTEA default_scorer_registry().

    Base-evidence scorers lead (canonical order = base evidence first), then
    knn_similarity / classifier, then the two priors.
    """
    assert list(SCORE_VECTOR_BY_SCORER) == [
        "alignment",
        "taxonomy",
        "label_embedding",
        "knn_similarity",
        "classifier",
        "self_prior",
        "association",
    ]


def test_nk_keeps_base_evidence_and_drops_only_the_two_priors() -> None:
    """NK has no pre-cutoff known terms, so only self_prior + association drop.

    The three base-evidence columns are NOT priors, so they survive on NK and
    restore the sequence / taxonomy / label-embedding signal a priors-only
    vector lost.
    """
    cols = combiner_columns_for_category("NK")
    for prior_col in _PRIOR_COLUMNS:
        assert prior_col not in cols
    # Base evidence + knn + classifier all apply to NK.
    assert cols == _BASE_EVIDENCE_COLUMNS + _KNN_CLF_COLUMNS


@pytest.mark.parametrize("cat", ["lk", "pk", "LK", "PK"])
def test_lk_pk_keep_full_vector(cat: str) -> None:
    """LK / PK keep the priors (case-insensitive)."""
    assert combiner_columns_for_category(cat) == DEFAULT_COMBINER_COLUMNS


def test_resolve_uses_cell_prefix_for_category() -> None:
    """resolve_combiner_columns reads the category from the cell prefix."""
    assert resolve_combiner_columns("nk-bpo") == combiner_columns_for_category("nk")
    assert resolve_combiner_columns("pk-cco") == DEFAULT_COMBINER_COLUMNS


def test_explicit_columns_win_verbatim_and_dedup() -> None:
    """Operator-supplied columns override the default, order-preserving + deduped."""
    out = resolve_combiner_columns(
        "nk-bpo", ["classifier_score", "distance", "classifier_score"]
    )
    assert out == ["classifier_score", "distance"]


# ---------------------------------------------------------------------------
# TrainConfig.feature_override
# ---------------------------------------------------------------------------


def test_feature_override_restricts_selected_features() -> None:
    """feature_override makes selected_features return exactly the given columns."""
    cfg = TrainConfig(feature_override=["distance", "classifier_score"])
    assert cfg.selected_features() == ["distance", "classifier_score"]


def test_feature_override_still_honours_drop_features() -> None:
    """An ablation via drop_features removes one score from the override vector."""
    cfg = TrainConfig(
        feature_override=["distance", "classifier_score"],
        drop_features=["classifier_score"],
    )
    assert cfg.selected_features() == ["distance"]


def test_feature_override_none_keeps_monolith_features() -> None:
    """Without feature_override the monolith uses the lab default feature set.

    Asserted against DEFAULT_TRAINING_FEATURES rather than ALL_FEATURES. The
    contracts catalogue includes families the lab has not adopted, so
    comparing against it would make this test tautological: it would follow
    the catalogue wherever a dependency bump moved it.
    """
    from protea_reranker_lab.contracts import DEFAULT_TRAINING_FEATURES

    cfg = TrainConfig()
    assert set(cfg.selected_features()) == set(DEFAULT_TRAINING_FEATURES)


# ---------------------------------------------------------------------------
# CLI wiring (_build_spec)
# ---------------------------------------------------------------------------


def _spec_for(argv: list[str]):
    return _build_spec(_parse_args(argv + ["--wandb-mode", "disabled"]))


def test_cli_combiner_flag_sets_override_per_category() -> None:
    spec = _spec_for(["--dataset", "ds", "--cell", "nk-bpo", "--combiner"])
    assert spec.model.defaults["feature_override"] == combiner_columns_for_category("nk")


def test_cli_combiner_columns_implies_combiner_and_wins() -> None:
    spec = _spec_for([
        "--dataset", "ds", "--cell", "pk-bpo",
        "--combiner-columns", "distance,classifier_score",
    ])
    assert spec.model.defaults["feature_override"] == ["distance", "classifier_score"]


def test_cli_combiner_shrinks_num_leaves_by_default() -> None:
    spec = _spec_for(["--dataset", "ds", "--cell", "lk-bpo", "--combiner"])
    assert spec.model.defaults["num_leaves"] == _COMBINER_NUM_LEAVES_DEFAULT


def test_cli_combiner_respects_explicit_num_leaves() -> None:
    spec = _spec_for([
        "--dataset", "ds", "--cell", "lk-bpo", "--combiner", "--num-leaves", "31",
    ])
    assert spec.model.defaults["num_leaves"] == 31


def test_cli_monolith_unchanged_no_override_default_leaves() -> None:
    """Without --combiner the spec carries no override and the monolith leaf count."""
    spec = _spec_for(["--dataset", "ds", "--cell", "lk-bpo"])
    assert "feature_override" not in spec.model.defaults
    assert spec.model.defaults["num_leaves"] == _MONOLITH_NUM_LEAVES_DEFAULT


def test_cli_combiner_defaults_objective_to_binary() -> None:
    """Combiner mode emits calibrated [0,1] probabilities for the threshold sweep.

    f_micro_w applies a GLOBAL threshold sweep, which wants calibrated
    probabilities, not the relative ranking scores lambdarank produces.
    """
    spec = _spec_for(["--dataset", "ds", "--cell", "pk-bpo", "--combiner"])
    assert spec.model.defaults["objective"] == "binary"


def test_cli_combiner_respects_explicit_objective() -> None:
    """An explicit --objective wins over the combiner binary default."""
    spec = _spec_for([
        "--dataset", "ds", "--cell", "pk-bpo", "--combiner",
        "--objective", "lambdarank",
    ])
    assert spec.model.defaults["objective"] == "lambdarank"


def test_cli_monolith_keeps_lambdarank_objective_default() -> None:
    """Without --combiner the monolith keeps its lambdarank ranking objective."""
    spec = _spec_for(["--dataset", "ds", "--cell", "lk-bpo"])
    assert spec.model.defaults["objective"] == "lambdarank"


def test_cli_combiner_defaults_neg_pos_ratio_to_fifty() -> None:
    """Combiner mode downsamples negatives to 50x positives when left unset."""
    spec = _spec_for(["--dataset", "ds", "--cell", "pk-bpo", "--combiner"])
    assert spec.model.defaults["neg_pos_ratio"] == _COMBINER_NEG_POS_RATIO_DEFAULT
    assert spec.training.neg_pos_ratio == _COMBINER_NEG_POS_RATIO_DEFAULT


def test_cli_combiner_respects_explicit_neg_pos_ratio() -> None:
    """An explicit --neg-pos-ratio wins over the combiner default."""
    spec = _spec_for([
        "--dataset", "ds", "--cell", "pk-bpo", "--combiner", "--neg-pos-ratio", "10",
    ])
    assert spec.model.defaults["neg_pos_ratio"] == 10.0
    assert spec.training.neg_pos_ratio == 10.0


def test_cli_combiner_explicit_none_neg_pos_ratio_disables_downsampling() -> None:
    """An explicit --neg-pos-ratio none keeps all negatives even in combiner mode."""
    spec = _spec_for([
        "--dataset", "ds", "--cell", "pk-bpo", "--combiner", "--neg-pos-ratio", "none",
    ])
    assert spec.model.defaults["neg_pos_ratio"] is None
    assert spec.training.neg_pos_ratio is None


def test_cli_monolith_keeps_none_neg_pos_ratio_default() -> None:
    """The monolith path keeps its None default (no downsampling)."""
    spec = _spec_for(["--dataset", "ds", "--cell", "lk-bpo"])
    assert spec.model.defaults["neg_pos_ratio"] is None
    assert spec.training.neg_pos_ratio is None


# ---------------------------------------------------------------------------
# End to end on a synthetic parquet (small; real LightGBM, no DB/MinIO)
# ---------------------------------------------------------------------------


_SCORE_COLS = DEFAULT_COMBINER_COLUMNS
# A monolith-only feature that must NOT leak into the combiner booster.
_NOISE_COL = "length_query"


def _write_synthetic(path: Path, *, category: str, n_proteins: int, seed: int) -> None:
    """Write a tiny labelled parquet carrying the score-vector + a noise column.

    The label is correlated with the classifier_score so a shallow booster can
    actually learn a non-trivial split (keeps the smoke fit meaningful).
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, list] = {c: [] for c in (
        "protein_accession", "label", "category", "aspect", "snapshot_pair",
        *_SCORE_COLS, _NOISE_COL,
    )}
    for pidx in range(n_proteins):
        acc = f"P{pidx:05d}"
        for _ in range(6):
            clf = float(rng.random())
            label = int(clf > 0.5 and rng.random() < 0.8)
            cols["protein_accession"].append(acc)
            cols["label"].append(label)
            cols["category"].append(category)
            cols["aspect"].append("bpo")
            cols["snapshot_pair"].append("v226-v230")
            cols["alignment_score_sw"].append(float(rng.random()))
            cols["taxonomic_distance"].append(float(rng.random()))
            cols["anc2vec_neighbor_maxcos"].append(float(rng.random()))
            cols["distance"].append(float(rng.random()))
            cols["neighbor_vote_fraction"].append(float(rng.random()))
            cols["classifier_score"].append(clf)
            cols["self_prior_score"].append(float(rng.random() < 0.2))
            cols["association_total"].append(float(rng.random()))
            cols["association_cross"].append(float(rng.random()))
            cols[_NOISE_COL].append(float(rng.integers(100, 900)))
    table = pa.table({
        "protein_accession": pa.array(cols["protein_accession"], pa.string()),
        "label": pa.array(cols["label"], pa.int8()),
        "category": pa.array(cols["category"], pa.string()),
        "aspect": pa.array(cols["aspect"], pa.string()),
        "snapshot_pair": pa.array(cols["snapshot_pair"], pa.string()),
        **{c: pa.array(cols[c], pa.float32()) for c in (*_SCORE_COLS, _NOISE_COL)},
    })
    pq.write_table(table, str(path))


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "combiner_ds"
    ds.mkdir()
    _write_synthetic(ds / "train.parquet", category="lk", n_proteins=60, seed=1)
    _write_synthetic(ds / "eval.parquet", category="lk", n_proteins=24, seed=2)
    manifest = {
        "schema_version": "v1",
        "name": "combiner-smoke",
        "k": 5,
        "embedding_config_id": "test-emb",
        "ontology_snapshot_id": "test-obo",
        "train_snapshot_pairs": ["v226-v230"],
        "eval_snapshot_pair": "v226-v230",
        "schema_sha": "deadbeef",
    }
    (ds / "manifest.json").write_text(json.dumps(manifest))
    return ds


def _run_combiner(ds: Path, out_dir: Path, extra: list[str] | None = None) -> dict:
    from protea_reranker_lab.runner import run_experiment

    argv = [
        "--dataset", str(ds), "--cell", "lk-bpo",
        "--objective", "lambdarank",
        "--num-boost-round", "20", "--early-stopping-rounds", "5",
        "--val-strategy", "protein_group", "--val-fraction", "0.25",
        "--wandb-mode", "disabled", "--combiner",
        "--output-dir", str(out_dir), *(extra or []),
    ]
    spec = _build_spec(_parse_args(argv))
    return run_experiment(spec, datasets_root=str(ds.parent))


def test_combiner_end_to_end_emits_registerable_artifacts(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "run"
    report = _run_combiner(synthetic_dataset, out_dir)

    assert report["status"] == "ok"
    # The three artifacts PROTEA import-by-reference consumes.
    assert (out_dir / "model.txt").exists()
    assert (out_dir / "spec.yaml").exists()
    assert (out_dir / "run.json").exists()


def test_combiner_booster_feature_names_are_exactly_the_score_vector(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """The booster trains over ONLY the score vector; the noise column is absent.

    PROTEA feeds the booster columns by name at inference (import-by-reference),
    so the baked feature_name list IS the contract that keeps the combiner
    consuming scores, not the 73-feature matrix.
    """
    out_dir = tmp_path / "run"
    _run_combiner(synthetic_dataset, out_dir)
    booster = lgb.Booster(model_file=str(out_dir / "model.txt"))
    names = booster.feature_name()
    assert names == DEFAULT_COMBINER_COLUMNS
    assert _NOISE_COL not in names


def test_combiner_run_json_records_provenance(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "run"
    _run_combiner(synthetic_dataset, out_dir)
    report = json.loads((out_dir / "run.json").read_text())
    feats = report["features"]
    assert feats["mode"] == "combiner"
    assert feats["combiner_score_columns"] == DEFAULT_COMBINER_COLUMNS
    assert feats["feature_columns"] == DEFAULT_COMBINER_COLUMNS
    assert feats["feature_count"] == len(DEFAULT_COMBINER_COLUMNS)


def test_combiner_columns_ablation_drops_score(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """--combiner-columns restricts the booster to a chosen 2-column vector."""
    out_dir = tmp_path / "run_abl"
    from protea_reranker_lab.runner import run_experiment

    argv = [
        "--dataset", str(synthetic_dataset), "--cell", "lk-bpo",
        "--num-boost-round", "20", "--early-stopping-rounds", "5",
        "--val-strategy", "protein_group", "--val-fraction", "0.25",
        "--wandb-mode", "disabled",
        "--combiner-columns", "distance,classifier_score",
        "--output-dir", str(out_dir),
    ]
    spec = _build_spec(_parse_args(argv))
    report = run_experiment(spec, datasets_root=str(synthetic_dataset.parent))
    assert report["status"] == "ok"
    booster = lgb.Booster(model_file=str(out_dir / "model.txt"))
    assert booster.feature_name() == ["distance", "classifier_score"]


def test_monolith_run_json_records_monolith_mode(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    """Monolith mode (no --combiner) still works and is tagged mode=monolith.

    The synthetic parquet only carries the score columns + one real contracts
    feature (length_query); the monolith zero-fills the rest, which is the
    historical behaviour. This guards that the combiner change did not alter the
    monolith feature-resolution path.
    """
    out_dir = tmp_path / "run_mono"
    from protea_reranker_lab.runner import run_experiment

    # Drop the categorical annotation_meta family so the smoke fit does not need
    # the categorical feature columns staged (the synthetic parquet carries only
    # numeric features + reserved columns). The monolith feature-resolution path
    # is exercised regardless; the categorical-staging path has its own tests.
    argv = [
        "--dataset", str(synthetic_dataset), "--cell", "lk-bpo",
        "--num-boost-round", "10", "--early-stopping-rounds", "5",
        "--val-strategy", "protein_group", "--val-fraction", "0.25",
        "--drop-feature-family", "annotation_meta",
        "--wandb-mode", "disabled", "--output-dir", str(out_dir),
    ]
    spec = _build_spec(_parse_args(argv))
    report = run_experiment(spec, datasets_root=str(synthetic_dataset.parent))
    assert report["status"] == "ok"
    feats = report["features"]
    assert feats["mode"] == "monolith"
    assert "combiner_score_columns" not in feats
    # Monolith uses the full contracts schema, far more than the 6 score columns.
    assert feats["feature_count"] > len(DEFAULT_COMBINER_COLUMNS)
