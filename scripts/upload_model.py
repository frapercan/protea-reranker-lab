"""Push a trained LightGBM booster back into PROTEA's ``reranker_model`` table.

The lab repo trains and evaluates boosters in isolation; once a configuration
wins a sweep, this script ships the serialized model + metrics to PROTEA so the
inference path (``rerank_predictions``) can pick it up by name/cell.

Usage:

    PROTEA_DB_URL=postgresql+psycopg://user:pass@host/db \\
    python scripts/upload_model.py \\
        --model runs/pk-bpo_lambdarank.txt \\
        --name pk-bpo_lambdarank_v1 \\
        --category pk --aspect bpo \\
        --metrics runs/pk-bpo_metrics.json \\
        --feature-importance runs/pk-bpo_fi.json \\
        [--prediction-set <uuid>] [--evaluation-set <uuid>]
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to LightGBM booster .txt")
    p.add_argument("--name", required=True, help="unique model name")
    p.add_argument("--category", required=True, choices=["nk", "lk", "pk"])
    p.add_argument("--aspect", default=None, choices=["bpo", "mfo", "cco", None])
    p.add_argument("--metrics", default=None, help="path to JSON metrics blob")
    p.add_argument("--feature-importance", default=None, help="path to JSON FI blob")
    p.add_argument("--prediction-set", default=None)
    p.add_argument("--evaluation-set", default=None)
    p.add_argument("--db-url", default=os.environ.get("PROTEA_DB_URL"))
    p.add_argument("--overwrite", action="store_true",
                   help="if a row with --name exists, replace it")
    return p.parse_args()


def _read_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def main() -> None:
    a = _args()
    if not a.db_url:
        raise SystemExit("PROTEA_DB_URL env var or --db-url flag required")

    model_data = Path(a.model).read_text()
    metrics = _read_json(a.metrics)
    feature_importance = _read_json(a.feature_importance)

    engine = create_engine(a.db_url)
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM reranker_model WHERE name = :n"),
            {"n": a.name},
        ).scalar_one_or_none()
        if existing and not a.overwrite:
            raise SystemExit(f"reranker_model '{a.name}' already exists ({existing}). Use --overwrite.")

        if existing:
            conn.execute(
                text("""
                    UPDATE reranker_model
                       SET model_data = :data,
                           metrics = CAST(:metrics AS JSONB),
                           feature_importance = CAST(:fi AS JSONB),
                           category = :category,
                           aspect = :aspect,
                           prediction_set_id = :ps,
                           evaluation_set_id = :es
                     WHERE id = :id
                """),
                {
                    "data": model_data,
                    "metrics": json.dumps(metrics),
                    "fi": json.dumps(feature_importance),
                    "category": a.category,
                    "aspect": a.aspect,
                    "ps": a.prediction_set,
                    "es": a.evaluation_set,
                    "id": existing,
                },
            )
            model_id = existing
        else:
            model_id = uuid.uuid4()
            conn.execute(
                text("""
                    INSERT INTO reranker_model
                        (id, name, category, aspect, model_data, metrics, feature_importance,
                         prediction_set_id, evaluation_set_id)
                    VALUES
                        (:id, :name, :category, :aspect, :data,
                         CAST(:metrics AS JSONB), CAST(:fi AS JSONB), :ps, :es)
                """),
                {
                    "id": model_id,
                    "name": a.name,
                    "category": a.category,
                    "aspect": a.aspect,
                    "data": model_data,
                    "metrics": json.dumps(metrics),
                    "fi": json.dumps(feature_importance),
                    "ps": a.prediction_set,
                    "es": a.evaluation_set,
                },
            )

    print(f"[upload] reranker_model id={model_id}  name={a.name}  "
          f"cell={a.category}-{a.aspect or '*'}  bytes={len(model_data):,}")


if __name__ == "__main__":
    main()
