#!/usr/bin/env python
"""One-shot exporter of the GO is_a/part_of parent map for a dataset.

Connects to PROTEA's Postgres, reads the ``go_term_relationship`` rows for the
``ontology_snapshot_id`` of a derived dataset's ``manifest.json``, and writes a
JSON ``{child_go_id: [parent_go_id, ...]}`` next to the dataset.

The lab's :mod:`staging` module reads that JSON to propagate labels to all
ancestors (CAFA True-Path-Rule). Decoupling the export keeps the lab DB-free.

Run with the PROTEA venv (which has ``psycopg``), not the lab venv. Example:

    /home/.cache/pypoetry/virtualenvs/protea-*/bin/python \\
        scripts/export_parent_map.py \\
        --manifest datasets/bench-v1-K5/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path,
                   help="dataset manifest.json (provides ontology_snapshot_id)")
    p.add_argument("--db-url", default="postgresql://protea:protea@localhost:5432/protea",
                   help="PROTEA Postgres URL")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path (default: <manifest_dir>/parent_map.json)")
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    snapshot_id = manifest["ontology_snapshot_id"]
    out_path = args.out or args.manifest.parent / "parent_map.json"

    try:
        import psycopg
    except ImportError:
        print("[err] psycopg not available — run this script with the PROTEA venv",
              file=sys.stderr)
        return 2

    sql = (
        "SELECT c.go_id AS child, p.go_id AS parent "
        "FROM go_term_relationship r "
        "JOIN go_term c ON c.id = r.child_go_term_id "
        "JOIN go_term p ON p.id = r.parent_go_term_id "
        "WHERE r.ontology_snapshot_id = %s "
        "AND r.relation_type IN ('is_a', 'part_of')"
    )

    with psycopg.connect(args.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (snapshot_id,))
            rows = cur.fetchall()

    parent_map: dict[str, list[str]] = {}
    for child, parent in rows:
        parent_map.setdefault(str(child), []).append(str(parent))

    out_path.write_text(json.dumps({
        "ontology_snapshot_id": snapshot_id,
        "n_edges": len(rows),
        "n_terms": len(parent_map),
        "parents": parent_map,
    }, indent=2))
    print(f"[parent_map] wrote {out_path}  edges={len(rows)}  terms={len(parent_map)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
