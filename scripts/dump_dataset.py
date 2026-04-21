"""Dump a PROTEA prediction_set + evaluation_set into a parquet dataset.

This script only handles the EVAL-side dump: it joins ``go_prediction`` rows
from an existing prediction_set against the NK/LK/PK ground truth from an
evaluation_set, writes one row per (protein, candidate GO term) with the
full feature vector and a binary label.

The TRAIN-side dump (multisnap deltas, per-cell splits) still requires a
PROTEA-side hook — ``train_reranker_auto --dump-only`` — because the
intermediate per-delta feature tables are never persisted by the current
pipeline. That hook is a TODO on the PROTEA repo.

Usage:

    PROTEA_DB_URL=postgresql+psycopg://user:pass@host/db \\
    python scripts/dump_dataset.py \\
        --prediction-set 4b734d30-29b3-48ce-a1f7-e9cf3b57156d \\
        --evaluation-set a73cb77c-9adf-4d55-b61f-c0b1bd05be01 \\
        --out datasets/bench-v1-K5/eval.parquet \\
        --manifest datasets/bench-v1-K5/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from protea_reranker_lab.reranker import ALL_FEATURES


NS_TO_ASPECT = {
    "biological_process": "bpo",
    "molecular_function": "mfo",
    "cellular_component": "cco",
    "BPO": "bpo", "MFO": "mfo", "CCO": "cco",
    "P": "bpo", "F": "mfo", "C": "cco",
}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-set", required=True)
    p.add_argument("--evaluation-set", required=True)
    p.add_argument("--out", required=True, help="path to eval.parquet")
    p.add_argument("--manifest", default=None, help="path to manifest.json (optional)")
    p.add_argument("--db-url", default=os.environ.get("PROTEA_DB_URL"))
    return p.parse_args()


def _build_labels_query() -> str:
    return """
    WITH targets AS (
        SELECT esp.protein_accession,
               esp.category,
               UNNEST(esp.true_go_ids_bpo) AS go_id,
               'bpo' AS aspect
        FROM evaluation_set_protein esp
        WHERE esp.evaluation_set_id = :eval_id
        UNION ALL
        SELECT esp.protein_accession, esp.category,
               UNNEST(esp.true_go_ids_mfo) AS go_id, 'mfo'
        FROM evaluation_set_protein esp
        WHERE esp.evaluation_set_id = :eval_id
        UNION ALL
        SELECT esp.protein_accession, esp.category,
               UNNEST(esp.true_go_ids_cco) AS go_id, 'cco'
        FROM evaluation_set_protein esp
        WHERE esp.evaluation_set_id = :eval_id
    )
    SELECT protein_accession, LOWER(category) AS category, aspect, go_id
    FROM targets
    """


def dump_eval(db_url: str, prediction_set: str, evaluation_set: str, out_path: Path) -> dict:
    engine = create_engine(db_url)
    feat_cols_sql = ", ".join(f"gp.{f}" for f in ALL_FEATURES if f != "aspect")
    # aspect comes from GOTerm; qualifier/evidence_code/taxonomic_relation are on gp already
    query = f"""
        SELECT
            gp.protein_accession,
            gt.go_id AS go_term_id,
            LOWER(gt.aspect) AS aspect,
            esp.category AS category,
            {feat_cols_sql}
        FROM go_prediction gp
        JOIN go_term gt ON gt.id = gp.go_term_id
        JOIN evaluation_set_protein esp
          ON esp.protein_accession = gp.protein_accession
         AND esp.evaluation_set_id = :eval_id
        WHERE gp.prediction_set_id = :ps_id
    """
    print(f"[dump] querying predictions…")
    with engine.connect() as conn:
        preds = pd.read_sql(text(query), conn, params={
            "ps_id": prediction_set, "eval_id": evaluation_set,
        })
        print(f"[dump] {len(preds):,} candidate predictions fetched")
        labels = pd.read_sql(text(_build_labels_query()), conn,
                             params={"eval_id": evaluation_set})
        print(f"[dump] {len(labels):,} positive labels fetched")

    labels["label"] = 1
    preds["category"] = preds["category"].str.lower()
    merged = preds.merge(
        labels[["protein_accession", "category", "aspect", "go_id", "label"]],
        how="left",
        left_on=["protein_accession", "category", "aspect", "go_term_id"],
        right_on=["protein_accession", "category", "aspect", "go_id"],
    )
    merged["label"] = merged["label"].fillna(0).astype("int8")
    merged = merged.drop(columns=["go_id"])
    merged["snapshot_pair"] = "eval"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    reserved = ["protein_accession", "go_term_id", "label", "category", "aspect", "snapshot_pair"]
    ordered = reserved + [f for f in ALL_FEATURES if f in merged.columns]
    merged = merged[ordered]
    print(f"[dump] writing {out_path} ({len(merged):,} rows, {len(ordered)} cols)")
    merged.to_parquet(out_path, index=False, compression="snappy")

    stats = {
        "n_rows": len(merged),
        "n_positives": int(merged["label"].sum()),
        "cells": sorted(set(zip(merged["category"], merged["aspect"]))),
    }
    return stats


def schema_sha() -> str:
    payload = json.dumps(ALL_FEATURES, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> None:
    a = _args()
    if not a.db_url:
        raise SystemExit("PROTEA_DB_URL env var or --db-url flag required")
    stats = dump_eval(a.db_url, a.prediction_set, a.evaluation_set, Path(a.out))
    print(f"[dump] {stats['n_positives']:,} positives / {stats['n_rows']:,} rows")
    if a.manifest:
        manifest = {
            "name": Path(a.out).parent.name,
            "k": None,  # fill from prediction_set.limit_per_entry if needed
            "embedding_config_id": None,
            "ontology_snapshot_id": None,
            "train_snapshot_pairs": [],
            "eval_snapshot_pair": "eval",
            "schema_sha": schema_sha(),
            "n_eval_rows": stats["n_rows"],
        }
        Path(a.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"[dump] wrote {a.manifest}")


if __name__ == "__main__":
    main()
