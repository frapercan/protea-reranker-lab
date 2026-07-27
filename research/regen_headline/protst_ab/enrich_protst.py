"""Enrich a sealed 227->230 parquet with the three protst_text columns by
running the EXACT deployed producer recipe (protea...._protst_text), per
snapshot pair with that pair's pre-cutoff t0 annotation set (the same t0 the
platform export stamps). Stamp semantics are byte-faithful to
``_protst_text._stamp_query``:

  protst_vote_fraction : fraction.get(go,0.0) for COVERED queries, NaN otherwise
  protst_present       : 1.0 if frac>0 else 0.0 for covered, NaN otherwise
  protst_text_score    : score[go] if the term drew a vote, else NaN

The kNN vote itself is a batched-BLAS replica of ``_knn_vote`` validated to
match the producer exactly (see enrich smoke test). numpy only, never pgvector.

Usage: python enrich_protst.py <in.parquet> <out.parquet> [--eval-only]
Run with the deploy venv (needs protea + DB + minio-free).
"""
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from protea.core.operations.predict_go_terms import _protst_text as P
from protea.infrastructure.session import build_session_factory, session_scope
from protea.infrastructure.settings import load_settings

IN = Path(sys.argv[1])
OUT = Path(sys.argv[2])

# v_old of each snapshot pair -> goa AnnotationSet id (t0). Eval pair v227-v230
# stamps t0=v227. Verified against annotation_set(source='goa').
VOLD_SET = {
    "v160-v165": "a8e2ffd4-721c-493f-882b-db6b92cb98bd",
    "v165-v170": "d09b2b14-0b27-47d0-afd9-8c9e5c3c3052",
    "v170-v175": "f720c3c4-abc0-49f7-9d3c-bb917a77089c",
    "v175-v180": "eb33207a-d876-420e-be67-8f0bd2748909",
    "v180-v185": "2a5f3325-e4e8-4f6f-8f0c-82b1f6ca755f",
    "v185-v190": "55999285-e523-4388-999d-fe934cc09994",
    "v190-v195": "120edbcf-467a-4a76-8c36-a86cc51bed3d",
    "v195-v200": "5db5ea46-4a37-4f37-a282-5a97e98207c6",
    "v200-v205": "9493bc57-90d6-4929-ac02-f897af0c04e6",
    "v205-v211": "3c51a9c8-d7a2-40c5-86a3-30d659542e73",
    "v211-v215": "22cb2901-09ca-49fa-8ec9-271d3feda62e",
    "v215-v220": "b66eb37f-9e9d-4ee5-a6cd-c5379abbae30",
    "v220-v225": "1559d9f7-195d-4892-af16-8b58f7fc9942",
    "v225-v227": "38280963-c8cd-4a2b-95c9-fa682ecc232d",
    "v227-v230": "c905dffa-a5ce-430b-b17b-503e88666adb",
}
K = P.K_NEIGHBORS
CHUNK = 256  # query columns per BLAS matmul


def vote_group(sess, cfg, t0_set, proteins, prot_rows, gos, aspects, score, frac, pres):
    """Stamp score/frac/pres arrays in place for all rows of one t0 group.

    proteins: unique covered-or-not accessions of the group.
    prot_rows: dict acc -> list[row_idx] (rows of this group).
    gos/aspects: full-length row->go_id / row->aspect arrays.
    """
    bank = P._get_reference_bank(sess, cfg, t0_set)
    accs, mat, refgo = bank
    if not accs:
        return 0, 0, 0
    acc_index = {a: i for i, a in enumerate(accs)}
    qemb = P._load_query_embeddings(sess, cfg, proteins)
    covered_accs = [a for a in proteins if a in qemb]
    n_cov = 0
    n_score_stamp = 0
    n_bp_score = 0
    for start in range(0, len(covered_accs), CHUNK):
        chunk = covered_accs[start:start + CHUNK]
        Q = np.stack([qemb[a] for a in chunk], axis=1)          # (D, Mc)
        S = mat @ Q                                             # (N, Mc)
        for j, a in enumerate(chunk):
            sims = S[:, j].copy()
            si = acc_index.get(a)
            if si is not None:
                sims[si] = -np.inf
            n_elig = int(np.isfinite(sims).sum())
            if n_elig == 0:
                # still a covered query: stamp measured-zero frac/present
                for ridx in prot_rows[a]:
                    frac[ridx] = 0.0
                    pres[ridx] = 0.0
                n_cov += 1
                continue
            k_eff = min(K, n_elig)
            top = (np.argpartition(-sims, k_eff - 1)[:k_eff]
                   if k_eff < len(accs) else np.arange(len(accs)))
            votes = {}
            counts = {}
            for i in top:
                cos = float(sims[i])
                if cos <= 0.0 or not np.isfinite(cos):
                    continue
                for go in refgo.get(accs[i], ()):
                    votes[go] = votes.get(go, 0.0) + cos
                    counts[go] = counts.get(go, 0) + 1
            if votes:
                mx = max(votes.values())
                sc = {g: v / mx for g, v in votes.items()} if mx > 0.0 else {}
                fr = {g: c / k_eff for g, c in counts.items()}
            else:
                sc, fr = {}, {}
            n_cov += 1
            for ridx in prot_rows[a]:
                go = gos[ridx]
                f = fr.get(go, 0.0)
                frac[ridx] = f
                pres[ridx] = 1.0 if f > 0.0 else 0.0
                if go in sc:
                    score[ridx] = sc[go]
                    n_score_stamp += 1
                    if aspects[ridx] == "bpo":
                        n_bp_score += 1
    return n_cov, n_score_stamp, n_bp_score


def main():
    s = load_settings(Path("/home/frapercan/Thesis2/worktrees/protea-deploy"))
    factory = build_session_factory(s.db_url)
    cfg = P._resolve_protst_config_id(None)

    t = pq.read_table(str(IN), columns=["protein_accession", "go_term_id", "snapshot_pair", "aspect"])
    n = t.num_rows
    prots = t.column("protein_accession").to_pylist()
    gos = t.column("go_term_id").to_pylist()
    snaps = t.column("snapshot_pair").to_pylist()
    asps = t.column("aspect").to_pylist()
    print(f"rows={n}", flush=True)

    score = np.full(n, np.nan, dtype=np.float32)
    frac = np.full(n, np.nan, dtype=np.float32)
    pres = np.full(n, np.nan, dtype=np.float32)

    # group row indices by snapshot pair
    rows_by_pair = defaultdict(list)
    for i, sp in enumerate(snaps):
        rows_by_pair[sp].append(i)

    tot_cov = tot_score = tot_bp = 0
    with session_scope(factory) as sess:
        for sp in sorted(rows_by_pair):
            idxs = rows_by_pair[sp]
            t0_set = uuid.UUID(VOLD_SET[sp])
            prot_rows = defaultdict(list)
            for ridx in idxs:
                prot_rows[prots[ridx]].append(ridx)
            proteins = sorted(prot_rows)
            t0 = time.time()
            cov, sst, bp = vote_group(sess, cfg, t0_set, proteins, prot_rows, gos, asps, score, frac, pres)
            tot_cov += cov
            tot_score += sst
            tot_bp += bp
            print(f"[{sp}] rows={len(idxs)} proteins={len(proteins)} covered={cov} "
                  f"score_stamps={sst} bp_score_stamps={bp} t={time.time()-t0:.0f}s", flush=True)

    # histogram sanity on protst_text_score (BP rows)
    bp_mask = np.array([a == "bpo" for a in asps])
    bp_score = score[bp_mask]
    finite_bp = bp_score[np.isfinite(bp_score)]
    print(f"\nSUMMARY: covered_queries={tot_cov} score_stamps={tot_score} bp_score_stamps={tot_bp}")
    print(f"protst_text_score BP: n_rows={bp_mask.sum()} n_finite={finite_bp.size} "
          f"({100*finite_bp.size/max(1,bp_mask.sum()):.1f}%) "
          f"min={finite_bp.min() if finite_bp.size else 'NA'} "
          f"max={finite_bp.max() if finite_bp.size else 'NA'} "
          f"mean={finite_bp.mean() if finite_bp.size else 'NA'}")
    if finite_bp.size:
        qs = np.quantile(finite_bp, [0, .1, .25, .5, .75, .9, 1.0])
        print("  BP score quantiles [0,.1,.25,.5,.75,.9,1]:", np.round(qs, 4).tolist())
    pf = pres[np.isfinite(pres)]
    print(f"protst_present: covered_rows={pf.size} present1_frac={float((pf==1.0).mean()) if pf.size else 'NA'}")

    # stream-append the 3 columns to the full parquet
    pf_in = pq.ParquetFile(str(IN))
    writer = None
    off = 0
    for b in range(pf_in.num_row_groups):
        tb = pf_in.read_row_group(b)
        m = tb.num_rows
        tb = tb.append_column("protst_text_score", pa.array(score[off:off + m]))
        tb = tb.append_column("protst_vote_fraction", pa.array(frac[off:off + m]))
        tb = tb.append_column("protst_present", pa.array(pres[off:off + m]))
        if writer is None:
            writer = pq.ParquetWriter(str(OUT), tb.schema)
        writer.write_table(tb)
        off += m
    if writer:
        writer.close()
    print(f"\nwrote {OUT} rows_appended={off}", flush=True)


if __name__ == "__main__":
    main()
