"""Pull + cache PRODUCTION embeddings for the L10-std Step 3 regeneration.

Reads (read-only) the DB and caches two artefacts under step3/:
  - l10_prod.npz : ALL SequenceEmbedding rows for the L10 base config 81436dba
                   (accs + float32 array, single chunk per protein). Values are the
                   DB-stored L10 / 32 uniform scale; z-score absorbs the /32, read as-is.
  - champ_codes.npz : champion learned codes (query + ref) = champion head
                      (ankh_base_hardneg.pt) applied to its PRODUCTION L48 base
                      08234f06 -> l2n -> Linear -> top-k 128. The apples baseline.

Author: Francisco Miguel Perez Canales.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg2
import torch

W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline/step3")
DSN = "host=localhost dbname=protea user=protea password=protea"
L10_CFG = "81436dba-1324-4536-bccc-122ac45dd9ba"
CHAMP_BASE = "08234f06-ba76-4d7d-aaec-ae601096b4fa"
CHAMP_HEAD = "/home/frapercan/Thesis2/storage/learned_encoders/ankh_base_hardneg.pt"


def parse(text: str) -> np.ndarray:
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def l2n(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X: np.ndarray, k: int) -> np.ndarray:
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def pull_config_all(cur, cfg: str) -> tuple[list[str], np.ndarray]:
    """All single-chunk embeddings for a config -> (accs, (N,768) float32)."""
    named = "cur_" + cfg.replace("-", "")[:20]
    cur2 = cur.connection.cursor(name=named)
    cur2.itersize = 20000
    cur2.execute(
        """SELECT p.accession, se.embedding::text
             FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
            WHERE se.embedding_config_id = %s
            ORDER BY p.accession""", (cfg,))
    accs: list[str] = []
    vecs: list[np.ndarray] = []
    t0 = time.time()
    for acc, emb in cur2:
        accs.append(acc)
        vecs.append(parse(emb))
        if len(accs) % 100000 == 0:
            print(f"  pulled {len(accs):,} ({time.time()-t0:.0f}s)", flush=True)
    cur2.close()
    return accs, np.vstack(vecs).astype(np.float32)


def pull_subset(cur, accs: list[str], cfg: str) -> dict[str, np.ndarray]:
    cur.execute(
        """SELECT p.accession, se.embedding::text
             FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
            WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)
            ORDER BY p.accession, se.chunk_index_s""", (cfg, list(accs)))
    ch: dict[str, list] = {}
    for acc, emb in cur.fetchall():
        ch.setdefault(acc, []).append(parse(emb))
    return {a: np.vstack(v).mean(0).astype(np.float32) for a, v in ch.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = json.load(open(W / "emb_ankh_base" / "meta.json"))["accs"]
    r = json.load(open(W / "ref_emb" / "meta.json"))["accs"]

    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    # --- L10 production embeddings for the whole config ---
    l10_path = OUT / "l10_prod.npz"
    if not l10_path.exists():
        print("[pull] L10 base config 81436dba (all rows)...", flush=True)
        accs, arr = pull_config_all(cur, L10_CFG)
        np.savez(l10_path, accs=np.array(accs), emb=arr)
        print(f"  saved {l10_path} : {arr.shape}", flush=True)
    else:
        print(f"[skip] {l10_path} exists", flush=True)

    # --- champion codes: champion head over production L48 base 08234f06 ---
    champ_path = OUT / "champ_codes.npz"
    if not champ_path.exists():
        print("[pull] champion base 08234f06 for query + ref...", flush=True)
        qe = pull_subset(cur, q, CHAMP_BASE)
        re = pull_subset(cur, r, CHAMP_BASE)
        Xq = np.vstack([qe[a] for a in q])
        Xr = np.vstack([re[a] for a in r])
        ck = torch.load(CHAMP_HEAD, map_location="cpu", weights_only=False)
        Wt = ck["state_dict"]["weight"].numpy()
        b = ck["state_dict"]["bias"].numpy()
        k = int(ck["meta"]["top_k"])
        # champion recipe: raw L48 base -> l2n -> Linear -> top-k (NO z-score)
        Qc = topk_real(l2n(Xq) @ Wt.T + b, k)
        Rc = topk_real(l2n(Xr) @ Wt.T + b, k)
        # validate query codes against the pinned artefact
        saved = np.load(W / "query_d8979601.npy").astype(np.float32)
        diff = float(np.abs(Qc - saved).max())
        print(f"  champion query-code reproduction max abs diff vs pinned = {diff:.2e}", flush=True)
        np.savez(champ_path, q_codes=Qc.astype(np.float32), r_codes=Rc.astype(np.float32),
                 repro_maxdiff=np.array(diff))
        print(f"  saved {champ_path}", flush=True)
    else:
        print(f"[skip] {champ_path} exists", flush=True)

    cur.close()
    conn.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
