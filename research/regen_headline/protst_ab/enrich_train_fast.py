"""Efficient train-frame ProtST enrichment: load the full ProtST embedding bank
ONCE into a numpy matrix, then per snapshot-pair do only a cheap t0 GO-annotation
lookup and subset the in-memory matrix (no per-chunk DB embedding round-trips,
which was the IO bottleneck). Vote semantics are byte-identical to the deployed
producer ``_protst_text`` (validated: batched vote MATCHES _knn_vote exactly).

Per (query protein, candidate GO term): top-30 cosine neighbours over the t0
reference bank (proteins with a ProtST embedding AND >=1 t0 annotation),
cosine-weight vote of neighbours' t0 go_ids, per-query-max normalise, exclude the
query's own accession. Stamp semantics == _stamp_query:
  vote_fraction = frac.get(go,0.0) covered / NaN uncovered
  present       = 1.0 if frac>0 else 0.0 covered / NaN uncovered
  text_score    = score[go] if the term drew a vote else NaN
numpy only, never pgvector.
"""
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from protea.infrastructure.orm.models.annotation.go_term import GOTerm
from protea.infrastructure.orm.models.annotation.protein_go_annotation import ProteinGOAnnotation
from protea.infrastructure.orm.models.embedding.sequence_embedding import SequenceEmbedding
from protea.infrastructure.orm.models.protein.protein import Protein
from protea.infrastructure.session import build_session_factory, session_scope
from protea.infrastructure.settings import load_settings

IN = Path("/home/frapercan/Thesis2/storage/regen_headline/protst_ab/sealed_train.parquet")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline/protst_ab/enriched_train.parquet")
CFG = uuid.UUID("bd3cd470-e384-4f6a-90cf-574704419373")
K = 30
CHUNK = 256

# v_old of each train pair -> goa AnnotationSet id (t0). VAL pair v225-v227 -> t0=v225.
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
}


def l2(v):
    v = np.asarray(v, dtype=np.float32)
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm == 0.0:
        return None
    return v / nrm


def load_full_bank(sess):
    """acc -> row index, plus MAT (N,512) unit vectors. ONE join, no t0 filter."""
    t0 = time.time()
    q = (
        sess.query(Protein.accession, SequenceEmbedding.embedding)
        .join(
            SequenceEmbedding,
            (SequenceEmbedding.sequence_id == Protein.sequence_id)
            & (SequenceEmbedding.embedding_config_id == CFG)
            & (SequenceEmbedding.chunk_index_s == 0),
        )
        .yield_per(20000)
    )
    accs = []
    vecs = []
    for acc, emb in q:
        u = l2(emb.to_numpy())
        if u is None:
            continue
        accs.append(acc)
        vecs.append(u)
    mat = np.vstack(vecs).astype(np.float32)
    acc_row = {a: i for i, a in enumerate(accs)}
    print(f"full bank: N={len(accs)} dim={mat.shape} load={time.time()-t0:.0f}s", flush=True)
    return accs, mat, acc_row


def load_t0_go(sess, t0_set):
    """acc -> set(go_id) for one t0 annotation set. ONE query."""
    out = defaultdict(set)
    q = (
        sess.query(ProteinGOAnnotation.protein_accession, GOTerm.go_id)
        .join(GOTerm, GOTerm.id == ProteinGOAnnotation.go_term_id)
        .filter(ProteinGOAnnotation.annotation_set_id == t0_set)
        .yield_per(50000)
    )
    for acc, go in q:
        out[acc].add(go)
    return dict(out)


def vote_pair(full, t0_go, prot_rows, gos, asps, score, frac, pres):
    accs_all, mat_all, acc_row = full
    # eligible refs = t0-annotated accessions that have an embedding
    elig = [a for a in t0_go if a in acc_row]
    if not elig:
        return 0, 0, 0
    rows = np.fromiter((acc_row[a] for a in elig), dtype=np.int64, count=len(elig))
    M = mat_all[rows]                        # (Ne, 512)
    refgo = [t0_go[a] for a in elig]         # aligned to M rows
    elig_index = {a: i for i, a in enumerate(elig)}
    proteins = sorted(prot_rows)
    covered = [a for a in proteins if a in acc_row]  # query needs an embedding
    n_cov = n_score = n_bp = 0
    for start in range(0, len(covered), CHUNK):
        chunk = covered[start:start + CHUNK]
        Q = np.stack([mat_all[acc_row[a]] for a in chunk], axis=1)   # (512, Mc)
        S = M @ Q                                                    # (Ne, Mc)
        for j, a in enumerate(chunk):
            sims = S[:, j].copy()
            si = elig_index.get(a)            # leakage: exclude self if in bank
            if si is not None:
                sims[si] = -np.inf
            n_elig = int(np.isfinite(sims).sum())
            n_cov += 1
            if n_elig == 0:
                for ridx in prot_rows[a]:
                    frac[ridx] = 0.0
                    pres[ridx] = 0.0
                continue
            k_eff = min(K, n_elig)
            top = (np.argpartition(-sims, k_eff - 1)[:k_eff]
                   if k_eff < len(elig) else np.arange(len(elig)))
            votes = {}
            counts = {}
            for i in top:
                cos = float(sims[i])
                if cos <= 0.0 or not np.isfinite(cos):
                    continue
                for go in refgo[i]:
                    votes[go] = votes.get(go, 0.0) + cos
                    counts[go] = counts.get(go, 0) + 1
            if votes:
                mx = max(votes.values())
                sc = {g: v / mx for g, v in votes.items()} if mx > 0.0 else {}
                fr = {g: c / k_eff for g, c in counts.items()}
            else:
                sc, fr = {}, {}
            for ridx in prot_rows[a]:
                go = gos[ridx]
                f = fr.get(go, 0.0)
                frac[ridx] = f
                pres[ridx] = 1.0 if f > 0.0 else 0.0
                if go in sc:
                    score[ridx] = sc[go]
                    n_score += 1
                    if asps[ridx] == "bpo":
                        n_bp += 1
    return n_cov, n_score, n_bp


def main():
    s = load_settings(Path("/home/frapercan/Thesis2/worktrees/protea-deploy"))
    factory = build_session_factory(s.db_url)

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

    rows_by_pair = defaultdict(list)
    for i, sp in enumerate(snaps):
        rows_by_pair[sp].append(i)

    tot_cov = tot_score = tot_bp = 0
    with session_scope(factory) as sess:
        full = load_full_bank(sess)
        for sp in sorted(rows_by_pair):
            idxs = rows_by_pair[sp]
            t0_set = uuid.UUID(VOLD_SET[sp])
            t0 = time.time()
            t0_go = load_t0_go(sess, t0_set)
            prot_rows = defaultdict(list)
            for ridx in idxs:
                prot_rows[prots[ridx]].append(ridx)
            cov, sst, bp = vote_pair(full, t0_go, prot_rows, gos, asps, score, frac, pres)
            tot_cov += cov
            tot_score += sst
            tot_bp += bp
            print(f"[{sp}] rows={len(idxs)} proteins={len(prot_rows)} refs={len(t0_go)} "
                  f"covered={cov} score_stamps={sst} bp_score_stamps={bp} t={time.time()-t0:.0f}s", flush=True)

    bp_mask = np.array([a == "bpo" for a in asps])
    fb = score[bp_mask]
    fb = fb[np.isfinite(fb)]
    print(f"\nSUMMARY covered={tot_cov} score_stamps={tot_score} bp_score_stamps={tot_bp}")
    print(f"protst_text_score BP finite={fb.size}/{int(bp_mask.sum())} "
          f"({100*fb.size/max(1,int(bp_mask.sum())):.1f}%) "
          f"min={fb.min() if fb.size else 'NA'} mean={fb.mean() if fb.size else 'NA'} max={fb.max() if fb.size else 'NA'}", flush=True)

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
