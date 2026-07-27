"""Offline full-pool ProtST-ESM1b embedding extraction + DB load (Path B).

Author: Francisco Miguel Perez Canales

Materialises the ProtST text-aligned protein_feature (512-d) for the full v227
reference pool (every sequence carrying a champion 08234f06 embedding) plus the
acc27f47 eval queries, storing them as SequenceEmbedding rows under a real,
offline-load EmbeddingConfig. The runtime cannot compute this backend; the row
is a provenance label and these vectors are the pinned offline recipe
(byte-for-byte the storage/text_scorer/extract_protst.py forward).

ProtST protein_feature magnitudes are tiny (per-dim abs max ~0.57, L2 norm
~0.75..3.7), so the fp16 halfvec store is safe with embedding_scale=1.0 and no
normalisation: vectors are written raw.

Idempotent + resumable: on start it skips sequence_ids already stored under the
config, so re-launching after an interruption continues where it stopped.
"""

import os
import sys
import time

import numpy as np
import psycopg
import torch
from transformers import AutoModel, AutoTokenizer

CONFIG_ID = "594701e0-8fbb-4571-9f69-3d227eeaa3be"
CHAMPION_CONFIG_ID = "08234f06-ba76-4d7d-aaec-ae601096b4fa"
QUERY_SET_ID = "acc27f47-8d8f-4011-91be-f96246a0c977"
DSN = "host=localhost dbname=protea user=protea password=protea"

DIM = 512
MAXLEN = 1022
BATCH = 16
COMMIT_EVERY = 20  # batches between commits + progress print
MIN_FREE_GB = 20.0
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "extract.log")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


def free_gb(path=HERE):
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024**3)


def df_guard():
    gb = free_gb()
    if gb < MIN_FREE_GB:
        log(f"ABORT: disk free {gb:.1f}GB < {MIN_FREE_GB}GB guard")
        sys.exit(1)
    return gb


def load_targets(conn):
    """Return list of (sequence_id, sequence_text) still pending, sorted by length."""
    with conn.cursor() as cur:
        log("selecting target pool (champion pool UNION query set) ...")
        cur.execute(
            """
            SELECT s.id, s.sequence
            FROM sequence s
            WHERE s.id IN (
                SELECT sequence_id FROM sequence_embedding
                    WHERE embedding_config_id = %(champ)s
                UNION
                SELECT sequence_id FROM query_set_entry
                    WHERE query_set_id = %(qs)s
            )
            """,
            {"champ": CHAMPION_CONFIG_ID, "qs": QUERY_SET_ID},
        )
        targets = cur.fetchall()
        log(f"target pool size: {len(targets)}")

        cur.execute(
            "SELECT sequence_id FROM sequence_embedding WHERE embedding_config_id = %s",
            (CONFIG_ID,),
        )
        done = {r[0] for r in cur.fetchall()}
        log(f"already stored under ProtST config: {len(done)}")

    pending = [(sid, seq) for sid, seq in targets if sid not in done]
    # sort by length so batches pad minimally (throughput)
    pending.sort(key=lambda t: len(t[1]))
    log(f"pending to extract: {len(pending)}")
    return pending


def vec_to_halfvec_str(vec):
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


def forward_batch(model, tok, seqs):
    enc = tok(
        [s[:MAXLEN] for s in seqs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAXLEN + 2,
    ).to("cuda")
    with torch.no_grad():
        out = model.protein_model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )
    feat = out.protein_feature if hasattr(out, "protein_feature") else out["protein_feature"]
    return feat.float().cpu().numpy()


def forward_safe(model, tok, seqs):
    """Forward with an OOM fallback (halve until it fits) for the unattended run."""
    try:
        return forward_batch(model, tok, seqs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(seqs) == 1:
            log(f"WARN: single-sequence OOM (len={len(seqs[0])}); skipping via 1-vec zero-pad retry")
            raise
        mid = len(seqs) // 2
        log(f"CUDA OOM at batch={len(seqs)}; splitting into {mid}+{len(seqs) - mid}")
        top = forward_safe(model, tok, seqs[:mid])
        bot = forward_safe(model, tok, seqs[mid:])
        return np.vstack([top, bot])


def main():
    log("=" * 70)
    log("ProtST full-pool extraction START")
    log(f"config_id={CONFIG_ID} batch={BATCH} maxlen={MAXLEN}")
    log(f"disk free at start: {free_gb():.1f}GB")
    df_guard()

    conn = psycopg.connect(DSN, autocommit=False)
    pending = load_targets(conn)
    total = len(pending)
    if total == 0:
        log("nothing pending; extraction already complete. exiting.")
        conn.close()
        return

    log("loading tokenizer facebook/esm1b_t33_650M_UR50S ...")
    tok = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
    log("loading model mila-intel/ProtST-esm1b (trust_remote_code) ...")
    model = (
        AutoModel.from_pretrained("mila-intel/ProtST-esm1b", trust_remote_code=True)
        .eval()
        .to("cuda")
    )
    log("model loaded on cuda; beginning extraction")

    insert_sql = (
        "INSERT INTO sequence_embedding "
        "(sequence_id, embedding_config_id, embedding, embedding_dim, chunk_index_s) "
        "VALUES (%s, %s, %s::halfvec, %s, 0) "
        "ON CONFLICT (sequence_id, embedding_config_id, chunk_index_s) DO NOTHING"
    )

    t0 = time.time()
    done_ct = 0
    with conn.cursor() as cur:
        for bi, i in enumerate(range(0, total, BATCH)):
            chunk = pending[i : i + BATCH]
            sids = [c[0] for c in chunk]
            seqs = [c[1] for c in chunk]
            feats = forward_safe(model, tok, seqs)
            rows = [
                (sid, CONFIG_ID, vec_to_halfvec_str(feats[j]), DIM)
                for j, sid in enumerate(sids)
            ]
            cur.executemany(insert_sql, rows)
            done_ct += len(chunk)

            if bi % COMMIT_EVERY == 0:
                conn.commit()
                df_guard()
                elapsed = time.time() - t0
                rate = done_ct / elapsed if elapsed > 0 else 0.0
                remaining = total - done_ct
                eta_h = (remaining / rate / 3600.0) if rate > 0 else float("nan")
                log(
                    f"  {done_ct}/{total} ({100.0 * done_ct / total:.1f}%) "
                    f"rate={rate:.1f} seq/s eta={eta_h:.2f}h free={free_gb():.1f}GB"
                )
        conn.commit()

    elapsed = time.time() - t0
    log(f"DONE: inserted {done_ct} sequences in {elapsed / 3600.0:.2f}h")
    conn.close()


if __name__ == "__main__":
    main()
