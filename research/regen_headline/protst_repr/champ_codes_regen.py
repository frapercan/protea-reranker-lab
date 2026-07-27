"""Regenerate champ_codes.npz (the apples champion baseline) for the protst repr screen.

step3/champ_codes.npz was cleaned up; rebuild it exactly as pull_embeddings.py did:
champion head (ankh_base_hardneg.pt) over the PRODUCTION L48 base 08234f06 for query+ref,
raw base -> mean-over-chunks -> l2n -> Linear -> top-k 128 (NO z-score). Validate the query
codes against the pinned query_d8979601.npy. READ-ONLY DB.

Author: Francisco Miguel Perez Canales.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import psycopg2
import torch

W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline/protst_repr")
DSN = "host=localhost dbname=protea user=protea password=protea"
CHAMP_BASE = "08234f06-ba76-4d7d-aaec-ae601096b4fa"
CHAMP_HEAD = "/home/frapercan/Thesis2/storage/learned_encoders/ankh_base_hardneg.pt"


def parse(text): return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X, k):
    if k >= X.shape[1]: return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def pull_subset(cur, accs, cfg):
    ch = {}
    B = 5000
    for i in range(0, len(accs), B):
        cur.execute(
            """SELECT p.accession, se.embedding::text
                 FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
                WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)
                ORDER BY p.accession, se.chunk_index_s""", (cfg, accs[i:i + B]))
        for acc, emb in cur.fetchall():
            ch.setdefault(acc, []).append(parse(emb))
    return {a: np.vstack(v).mean(0).astype(np.float32) for a, v in ch.items()}


def main():
    q = json.load(open(W / "emb_ankh_base" / "meta.json"))["accs"]
    r = json.load(open(W / "ref_emb" / "meta.json"))["accs"]
    conn = psycopg2.connect(DSN); cur = conn.cursor()
    print(f"[pull] production L48 base 08234f06 for {len(q)} query + {len(r)} ref...", flush=True)
    qe = pull_subset(cur, q, CHAMP_BASE)
    re = pull_subset(cur, r, CHAMP_BASE)
    cur.close(); conn.close()
    print(f"  pulled query {len(qe)}/{len(q)}  ref {len(re)}/{len(r)}", flush=True)
    Xq = np.vstack([qe[a] for a in q]); Xr = np.vstack([re[a] for a in r])
    ck = torch.load(CHAMP_HEAD, map_location="cpu", weights_only=False)
    Wt = ck["state_dict"]["weight"].numpy(); b = ck["state_dict"]["bias"].numpy()
    k = int(ck["meta"]["top_k"])
    Qc = topk_real(l2n(Xq) @ Wt.T + b, k)
    Rc = topk_real(l2n(Xr) @ Wt.T + b, k)
    saved = np.load(W / "query_d8979601.npy").astype(np.float32)
    diff = float(np.abs(Qc - saved).max())
    print(f"  champion query-code repro max abs diff vs pinned = {diff:.2e}", flush=True)
    np.savez(OUT / "champ_codes.npz", q_codes=Qc.astype(np.float32),
             r_codes=Rc.astype(np.float32), repro_maxdiff=np.array(diff))
    print(f"  saved {OUT/'champ_codes.npz'}  (top_k={k})", flush=True)


if __name__ == "__main__":
    main()
