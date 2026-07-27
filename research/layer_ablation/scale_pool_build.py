"""Build the 100,000-protein training pool and its sequences, once, so every
at-scale arm trains on exactly the same substrate.

The apples-to-apples control proved that the crown's failure was data starvation: the
same head recipe on 100k proteins reaches mean9 f_micro_w 0.2197 against the served
champion's 0.2150, scored on the same 15k reference. That control used the LAST layer,
which the fixed-representation ablation calls the worst base. The open question is
whether the best single z-scored layer (L10-std) or a learned layer mix separates from it
once trained at scale, rather than on the starved 15k substrate where neither could.

This script only prepares the substrate. It selects the pool the same way the lab harness
does (annotated proteins at the v227 snapshot, excluding every evaluation protein), pins
it with a seed, and writes the accession list plus the sequences. The layer extraction and
the training are separate steps, so the expensive forward pass happens once.

Read-only against the database.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/protea-reranker-lab/src")
from protea_reranker_lab.encoder_ablation import EncoderAblationSpec  # noqa: E402

W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
POOL_N = 100_000
SEED = 42
ANNOTATION_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0


def main() -> None:
    qry = json.load(open(W / "emb_ankh_base/meta.json"))["accs"]
    ref = json.load(open(W / "ref_emb/meta.json"))["accs"]
    banned = set(qry) | set(ref)
    assert not (set(qry) & set(ref)), "query and reference overlap"

    spec = EncoderAblationSpec()
    conn = psycopg2.connect(spec.dsn)
    cur = conn.cursor()

    # Same selection the harness uses: annotated at t0, sampled by a stable hash stride.
    cur.execute(
        """SELECT DISTINCT protein_accession FROM protein_go_annotation
           WHERE annotation_set_id = %s AND (hashtextextended(protein_accession, 42) %% %s) = 0""",
        (ANNOTATION_SET, max(2, 556000 // (POOL_N * 2))))
    cand = [a for (a,) in cur.fetchall() if a not in banned]
    print(f"  candidates after excluding the {len(banned):,} evaluation proteins: {len(cand):,}",
          flush=True)
    assert len(cand) >= POOL_N, f"only {len(cand):,} candidates for a {POOL_N:,} pool"

    rng = np.random.default_rng(SEED)
    rng.shuffle(cand)
    pool = sorted(cand[:POOL_N])

    # Sequences. The extractor caps at 2048 residues, the same cap the cached embeddings use.
    cur.execute(
        """SELECT p.accession, s.sequence FROM protein p
           JOIN sequence s ON s.id = p.sequence_id
           WHERE p.accession = ANY(%s)""", (pool,))
    seqs = {a: s for a, s in cur.fetchall() if s}
    cur.close()
    conn.close()

    missing = [a for a in pool if a not in seqs]
    if missing:
        print(f"  WARNING {len(missing):,} pool proteins have no sequence; dropping them", flush=True)
        pool = [a for a in pool if a in seqs]

    (W / "scale_pool_seqs.json").write_text(json.dumps({a: seqs[a] for a in pool}))
    (W / "scale_pool_meta.json").write_text(json.dumps(
        {"accs": pool, "n": len(pool), "seed": SEED, "annotation_set": ANNOTATION_SET,
         "excluded_query": len(qry), "excluded_reference": len(ref)}, indent=2))
    lens = np.array([len(seqs[a]) for a in pool])
    print(f"  pool written: {len(pool):,} proteins | length median {int(np.median(lens))} "
          f"p95 {int(np.percentile(lens, 95))} max {int(lens.max())}", flush=True)
    print(f"  over the 2048 cap: {(lens > 2048).sum():,} ({100 * (lens > 2048).mean():.1f}%)",
          flush=True)


if __name__ == "__main__":
    main()
