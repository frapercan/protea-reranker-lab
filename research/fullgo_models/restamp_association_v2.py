"""Snapshot-consistent association re-stamp (fixes the snapshot-id mismatch).

The native export put candidates in snapshot 35c3ad67 (v227) but conditioned
association on per-pair t0 sets in OTHER snapshots, so the cooccurrence ids
never matched -> association=0. Here we use the v227 cooccurrence (c905dffa,
35c3ad67, the SAME reference the predict path uses) as the single reference,
and condition on each protein's actual t0 known terms mapped into 35c3ad67 by
go_id string. Result matches what predict computes (1:1 string<->int in
35c3ad67). Replicates _score_association_candidates exactly.

Writes datasets/fullgo-native-parity-SELECT-220-227-assocfix/.
"""
from __future__ import annotations

import io
import sys
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from sqlalchemy import select, text

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/PROTEA")
from protea.core.operations.predict_go_terms import PredictGOTermsBatchOperation  # noqa: E402
from protea.core.operations.predict_go_terms._post_knn_pipeline import (  # noqa: E402
    _load_own_exp_for_association,
)
from protea.infrastructure.orm.models.annotation.go_term import GOTerm  # noqa: E402
from protea.infrastructure.session import build_session_factory, session_scope  # noqa: E402

DB = "postgresql+psycopg://protea:protea@127.0.0.1:5432/protea"
ONTO = uuid.UUID("35c3ad67-3002-47db-8f71-eeed69d22ad6")  # export/candidate snapshot
V227 = uuid.UUID("c905dffa-a5ce-430b-b17b-503e88666adb")  # reference cooccurrence set
SRC = "datasets/fullgo-native-parity-SELECT-220-227"
DST = "datasets/fullgo-native-parity-SELECT-220-227-assocfix"
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


def load_reference(session):
    """v227 cooccurrence keyed by 35c3ad67 ints: cooc{k:{t:count}}, freq{k:f}."""
    freq = {}
    for tid, f in session.execute(
        text("select term_id, freq from term_frequency where annotation_set_id=:s"),
        {"s": str(V227)},
    ):
        freq[int(tid)] = int(f)
    cooc: dict[int, dict[int, int]] = {}
    n = 0
    for k, t, c in session.execute(
        text("select known_term_id, candidate_term_id, cooccurrence_count "
             "from term_cooccurrence where annotation_set_id=:s"),
        {"s": str(V227)},
    ):
        cooc.setdefault(int(k), {})[int(t)] = int(c)
        n += 1
    print(f"reference: cooc known={len(cooc)} pairs={n} freq={len(freq)}", flush=True)
    return cooc, freq


def t0_to_canon(session, t0_set_str, s2i):
    """Map a t0 set's GO int ids -> 35c3ad67 ints via go_id string."""
    snap = session.execute(
        text("select ontology_snapshot_id from annotation_set where id=:s"), {"s": t0_set_str}
    ).scalar()
    rows = session.execute(
        select(GOTerm.id, GOTerm.go_id).where(GOTerm.ontology_snapshot_id == snap)
    ).all()
    return {int(i): s2i[g] for i, g in rows if g in s2i}


def score_rows(accs, cands_int, known35, cooc, freq, aspect):
    """Replicate _score_association_candidates in 35c3ad67 int space."""
    n = len(accs)
    total = [0.0] * n
    cross = [0.0] * n
    present = [0.0] * n
    for i in range(n):
        t = cands_int[i]
        if t is None:
            continue
        known = known35.get(accs[i])
        if not known:
            continue
        ta = aspect.get(t, "")
        tot = 0.0
        cr = 0.0
        for k in known:
            f = freq.get(k, 0)
            if f <= 0:
                continue
            c = cooc.get(k, {}).get(t, 0)
            if c <= 0:
                continue
            p = c / f
            tot += p
            if aspect.get(k, "") != ta:
                cr += p
        if tot > 0.0:
            total[i] = tot
            cross[i] = cr
            present[i] = 1.0
    return total, cross, present


def process(fname, factory, op, s2i, cooc, freq, aspect):
    print(f"  {fname}: loading...", flush=True)
    raw = _C.get_object("protea", f"{SRC}/{fname}").read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    tmp = f"/home/frapercan/Thesis2/storage/fullgo_models/_v2_{fname}"
    writer = None
    nrows = 0
    known_cache: dict[str, dict] = {}
    canon_cache: dict[str, dict] = {}
    for b in pf.iter_batches(batch_size=2_000_000):
        t = pa.Table.from_batches([b])
        accs = t.column("protein_accession").to_pylist()
        gos = t.column("go_term_id").to_pylist()
        sps = t.column("snapshot_pair").to_pylist()
        cands_int = [s2i.get(g) for g in gos]
        # known terms (35c3ad67 ints) per protein, grouped by t0 set
        by_pair: dict[str, set] = {}
        for acc, sp in zip(accs, sps, strict=True):
            by_pair.setdefault(sp, set()).add(acc)
        known35: dict[str, set] = {}
        for sp, accset in by_pair.items():
            v_old = sp.split("-")[0].lstrip("v")
            t0 = VERSION_TO_SET[v_old]
            if t0 not in canon_cache:
                with session_scope(factory) as s:
                    canon_cache[t0] = t0_to_canon(s, t0, s2i)
            cmap = canon_cache[t0]
            with session_scope(factory) as s:
                oe = _load_own_exp_for_association(op, s, uuid.UUID(t0), sorted(accset))
            for acc, terms in oe.items():
                known35[acc] = {cmap[x] for x in terms if x in cmap}
        tot, cr, pr = score_rows(accs, cands_int, known35, cooc, freq, aspect)
        nz = sum(1 for v in tot if v > 0)
        print(f"    {fname} batch rows={len(accs)} assoc_nonzero={nz}", flush=True)
        for name, vals in (("association_total", tot), ("association_cross", cr),
                           ("association_present", pr)):
            t = t.set_column(t.column_names.index(name), name, pa.array(vals, pa.float64()))
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="snappy")
        writer.write_table(t)
        nrows += len(accs)
    writer.close()
    _C.fput_object("protea", f"{DST}/{fname}", tmp)
    import os
    os.remove(tmp)
    print(f"  {fname}: wrote {nrows} -> s3://protea/{DST}/{fname}", flush=True)


def main() -> int:
    factory = build_session_factory(DB)
    op = PredictGOTermsBatchOperation()
    with session_scope(factory) as s:
        rows = s.execute(
            select(GOTerm.go_id, GOTerm.id, GOTerm.aspect).where(GOTerm.ontology_snapshot_id == ONTO)
        ).all()
    s2i = {g: int(i) for g, i, _ in rows}
    aspect = {int(i): (a or "") for _, i, a in rows}
    print(f"s2i(35c3ad67)={len(s2i)} terms", flush=True)
    with session_scope(factory) as s:
        cooc, freq = load_reference(s)
    for fname in ("eval.parquet", "train.parquet"):
        process(fname, factory, op, s2i, cooc, freq, aspect)
    man = _C.get_object("protea", f"{SRC}/manifest.json").read()
    _C.put_object("protea", f"{DST}/manifest.json", io.BytesIO(man), length=len(man))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
