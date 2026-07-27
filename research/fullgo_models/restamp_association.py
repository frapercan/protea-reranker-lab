"""Re-stamp ONLY the association_* columns on the existing parity export
parquet, reusing the EXACT predict producer (apply_association) so the values
match what predict computes (parity). Avoids the 8h full re-export: KNN +
classifier + everything else in the parquet are already correct; only
association was zero (term_cooccurrence was missing for the per-pair t0 sets,
now built for v160..v220).

Per snapshot_pair (vOLD-vNEW) the association conditions on the t0 = vOLD set.
Reads each pair's rows, builds {protein_accession, go_term_id:int} dicts, calls
apply_association(op, session, version_to_set[vOLD], accs, dicts, noop), reads
back association_total/cross/present, writes a new parquet to MinIO.

Run with the PROTEA venv from repositories/PROTEA.
"""

from __future__ import annotations

import io
import sys
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from sqlalchemy import select

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA")
from protea.core.operations.predict_go_terms._post_knn_pipeline import apply_association  # noqa: E402
from protea.core.operations.predict_go_terms import PredictGOTermsBatchOperation  # noqa: E402
from protea.infrastructure.orm.models.annotation.go_term import GOTerm  # noqa: E402
from protea.infrastructure.session import build_session_factory, session_scope  # noqa: E402

DB = "postgresql+psycopg://protea:protea@127.0.0.1:5432/protea"
ONTO = "35c3ad67-3002-47db-8f71-eeed69d22ad6"
SRC = "datasets/fullgo-native-parity-SELECT-220-227"
DST = "datasets/fullgo-native-parity-SELECT-220-227-assocfix"
ASSOC_COLS = ["association_total", "association_cross", "association_present"]
VERSION_TO_SET = {
    "160": "a8e2ffd4-721c-493f-882b-db6b92cb98bd", "165": "d09b2b14-0b27-47d0-afd9-8c9e5c3c3052",
    "170": "f720c3c4-abc0-49f7-9d3c-bb917a77089c", "175": "eb33207a-d876-420e-be67-8f0bd2748909",
    "180": "2a5f3325-e4e8-4f6f-8f0c-82b1f6ca755f", "185": "55999285-e523-4388-999d-fe934cc09994",
    "190": "120edbcf-467a-4a76-8c36-a86cc51bed3d", "195": "5db5ea46-4a37-4f37-a282-5a97e98207c6",
    "200": "9493bc57-90d6-4929-ac02-f897af0c04e6", "205": "3c51a9c8-d7a2-40c5-86a3-30d659542e73",
    "211": "22cb2901-09ca-49fa-8ec9-271d3feda62e", "215": "b66eb37f-9e9d-4ee5-a6cd-c5379abbae30",
    "220": "1559d9f7-195d-4892-af16-8b58f7fc9942",
}
_C = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)


def _t0_set(snapshot_pair: str) -> uuid.UUID:
    v_old = snapshot_pair.split("-")[0].lstrip("v")
    return uuid.UUID(VERSION_TO_SET[v_old])


def _restamp_table(tbl, op, factory, str_to_int, fname) -> pa.Table:
    # only pull the 3 needed columns to python; the rest stay as arrow.
    accs_col = tbl.column("protein_accession").to_pylist()
    go_col = tbl.column("go_term_id").to_pylist()
    sp_col = tbl.column("snapshot_pair").to_pylist()
    n = tbl.num_rows
    by_pair: dict[str, list[int]] = {}
    for i, sp in enumerate(sp_col):
        by_pair.setdefault(sp, []).append(i)
    total = [0.0] * n
    cross = [0.0] * n
    present = [0.0] * n
    for sp, idxs in by_pair.items():
        recs, rec_idx = [], []
        for i in idxs:
            gid = str_to_int.get(go_col[i])
            if gid is None:
                continue
            recs.append({"protein_accession": accs_col[i], "go_term_id": gid})
            rec_idx.append(i)
        accs = sorted({r["protein_accession"] for r in recs})
        with session_scope(factory) as s:
            apply_association(op, s, _t0_set(sp), accs, recs, lambda *a, **k: None)
        for r, i in zip(recs, rec_idx, strict=True):
            total[i] = float(r.get("association_total", 0.0) or 0.0)
            cross[i] = float(r.get("association_cross", 0.0) or 0.0)
            present[i] = float(r.get("association_present", 0.0) or 0.0)
        nz = sum(1 for i in idxs if total[i] > 0)
        print(f"    {fname} pair {sp}: rows={len(idxs)} assoc_nonzero={nz}", flush=True)
    out = tbl
    for name, vals in (("association_total", total), ("association_cross", cross),
                       ("association_present", present)):
        out = out.set_column(out.column_names.index(name), name, pa.array(vals, pa.float64()))
    return out


def _process(fname, op, factory, str_to_int) -> None:
    print(f"  {fname}: loading...", flush=True)
    raw = _C.get_object("protea", f"{SRC}/{fname}").read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    tmp = f"/home/frapercan/Thesis2/storage/fullgo_models/_restamp_{fname}"
    writer = None
    nrows = 0
    for b in pf.iter_batches(batch_size=2_000_000):
        t = _restamp_table(pa.Table.from_batches([b]), op, factory, str_to_int, fname)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="snappy")
        writer.write_table(t)
        nrows += t.num_rows
    writer.close()
    _C.fput_object("protea", f"{DST}/{fname}", tmp)
    import os
    os.remove(tmp)
    print(f"  {fname}: wrote {nrows} rows -> s3://protea/{DST}/{fname}", flush=True)


def main() -> int:
    factory = build_session_factory(DB)
    op = PredictGOTermsBatchOperation()
    with session_scope(factory) as s:
        rows = s.execute(
            select(GOTerm.go_id, GOTerm.id).where(GOTerm.ontology_snapshot_id == uuid.UUID(ONTO))
        ).all()
    str_to_int = {go: gid for go, gid in rows}
    print(f"go str->int map: {len(str_to_int)} terms", flush=True)
    for fname in ("eval.parquet", "train.parquet"):
        _process(fname, op, factory, str_to_int)
    # copy manifest so the dataset is registerable
    man = _C.get_object("protea", f"{SRC}/manifest.json").read()
    _C.put_object("protea", f"{DST}/manifest.json", io.BytesIO(man), length=len(man))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
