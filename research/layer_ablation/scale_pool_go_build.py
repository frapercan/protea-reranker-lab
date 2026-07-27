"""Pull the GO leaves for the 100,000-protein training pool.

The at-scale layer-ablation head (crown_train.py, run against the pool base in
scale_pool_emb/) needs a Lin GO-semantic target for the pool proteins, exactly
as the 15k crown used ref_go.json for its reference. scale_pool_build.py pinned
the pool accessions and sequences but not their annotations, so this fills the
one missing prerequisite.

Read-only: a single SELECT joining protein_go_annotation to go_term at the same
t0 snapshot (GOA v227) the pool was sampled from, filtered to the pinned pool
accessions so membership is identical, never re-derived. Writes one artefact,
scale_pool_go.json, in ref_go.json's shape: {accession: [raw GO leaf, ...]}.
Closures are propagated at train time, not here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/protea-reranker-lab/src")
from protea_reranker_lab.encoder_ablation import EncoderAblationSpec  # noqa: E402

W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
ANNOTATION_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0 (matches scale_pool_build.py)


def main() -> None:
    accs = json.load(open(W / "scale_pool_meta.json"))["accs"]
    print(f"pool: {len(accs):,} accessions", flush=True)

    conn = psycopg2.connect(EncoderAblationSpec().dsn)
    cur = conn.cursor()
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id
             FROM protein_go_annotation pga
             JOIN go_term gt ON gt.id = pga.go_term_id
            WHERE pga.annotation_set_id = %s
              AND pga.protein_accession = ANY(%s)""",
        (ANNOTATION_SET, accs),
    )
    go: dict[str, list[str]] = {}
    for acc, go_id in cur:
        go.setdefault(acc, []).append(go_id)
    cur.close()
    conn.close()

    missing = [a for a in accs if a not in go]
    assert not missing, f"{len(missing)} pool proteins have no annotation (e.g. {missing[:3]})"
    total = sum(len(v) for v in go.values())
    print(f"  annotated: {len(go):,}/{len(accs):,} | {total:,} leaves | "
          f"mean {total / len(go):.1f}/protein", flush=True)

    out = W / "scale_pool_go.json"
    out.write_text(json.dumps(go))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
