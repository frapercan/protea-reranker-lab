#!/usr/bin/env python
"""SDR correlation readout STRATIFIED by protein length (T-CIENCIA, sparse.pdf gate).

The directional question, cheap and GPU-free: does the agreement of a SPARSE / chunk /
residue encoding with GO semantics beat (or close the gap to) dense-mean cosine
SPECIFICALLY on LONG proteins, where mean-pooling washes out per-domain signal?

This is the length-stratified version of ``run_sdr_a_correlation.py`` /
``run_sdr_chunk_correlation.py`` / ``run_sdr_c_residue.py``: it reuses the same
Spearman(arm_similarity, GO_semantic_similarity) readout but buckets the sampled
proteins by ``length(sequence)`` and reports one Spearman per (bucket x arm x metric).

Length buckets (from the real v227 distribution)::

    short      : <= 318
    medium     : 319 .. 969
    long       : 970 .. 1831
    very-long  : > 1831

Arms (density-matched; k swept in {32, 64, 128} where applicable):

  * ``dense-mean-cosine`` : raw ankh-base mean (08234f06, 768d) -> cosine        (baseline)
  * ``mean-SDR``          : k-WTA on the mean -> Tanimoto                         (A0)
  * ``chunk-SDR``         : per-chunk k-WTA -> top-k vote bundle (6542db1e) -> Tanimoto  (A1)
  * ``residue-SDR``       : per-residue top-k OR-bundle (npy sample) -> Tanimoto  (A2)
  * ``learned``           : mean-derived hard-neg k-WTA code (d8979601, 2048d) -> Tanimoto

The mean / chunk / learned arms share ONE unified DB sample (so their buckets are
directly comparable). The residue arm runs over the ~5000-protein per-residue npy
sample with its OWN bucketing (ACTUAL N reported per bucket; low-N buckets flagged).

Pairs are sampled WITHIN each bucket (same-bucket pairs only), so a bucket's Spearman
measures agreement among proteins of that length class.

Everything is leakage-clean (frozen v227 t0 pool + the t0 OBO) and READ-ONLY on the DB.
Logs to MLflow experiment ``sdr-length-stratified``.

Run inside the lab venv with MLflow live::

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python scripts/run_sdr_length_stratified.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from protea_reranker_lab.sdr import (
    GoDag,
    cosine_dense,
    information_content,
    kwta_active_set,
    kwta_binarise,
    lin_pairwise,
    propagate,
    resnik_pairwise,
    tanimoto_dense,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sdr-len] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdr-len")

EXPERIMENT = "sdr-length-stratified"

EMB_MEAN = "08234f06-ba76-4d7d-aaec-ae601096b4fa"  # ankh-base mean, 768d
EMB_CHUNK = "6542db1e-202a-4769-b933-2e0f85aa81e6"  # ankh-base per-chunk, 768d/chunk
EMB_LEARNED = "d8979601-ea59-4de1-9c16-21036ed67c36"  # learned hard-neg k-WTA code, 2048d
ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"

# (name, low_inclusive, high_inclusive); high=None means open-ended.
BUCKETS = [
    ("short", 0, 318),
    ("medium", 319, 969),
    ("long", 970, 1831),
    ("very-long", 1832, None),
]


def bucket_of(length: int) -> str:
    for name, lo, hi in BUCKETS:
        if length >= lo and (hi is None or length <= hi):
            return name
    return BUCKETS[-1][0]


def _parse_vec(text: str) -> np.ndarray:
    """Parse a pgvector text literal ``[a,b,c]`` into a float32 array."""
    return np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float32)


# ---------------------------------------------------------------------------
# Unified DB sample for the mean / chunk / learned arms (READ-ONLY)
# ---------------------------------------------------------------------------
def load_db_sample(dsn: str, n_target: int, seed: int, per_bucket_cap: int):
    """Load (acc, length, mean, chunk-list, learned) for a LENGTH-BALANCED sample.

    Returns dicts keyed by accession plus the ordered accession list. The mean and
    learned embeddings are dense rows; chunks are an (n_chunks, d) matrix; length is
    ``len(sequence)``.

    Long / very-long proteins are RARE (~3% / ~0.8% of the annotated pool), so a flat
    hash sample starves those buckets. We instead pull EVERY annotated protein's length
    once (cheap, indexed), bucket on the client, then cap each bucket at
    ``per_bucket_cap`` (short/medium are downsampled; long/very-long are taken whole).
    This keeps the long buckets at a usable N while bounding RAM.
    """
    import psycopg2

    rng = np.random.default_rng(seed)
    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True)
    cur = conn.cursor()

    log.info("scanning lengths for all v227-annotated proteins (for length-balanced cap)")
    cur.execute(
        """
        SELECT p.accession, p.sequence_id, length(s.sequence) AS seqlen
        FROM (SELECT DISTINCT protein_accession AS acc
              FROM protein_go_annotation WHERE annotation_set_id = %(ann)s) a
        JOIN protein p ON p.accession = a.acc
        JOIN sequence s ON s.id = p.sequence_id
        WHERE p.sequence_id IS NOT NULL
        """,
        {"ann": ANN_SET},
    )
    cand = cur.fetchall()
    log.info("  %d annotated proteins with a sequence", len(cand))

    acc_to_seq = {acc: sid for acc, sid, _ in cand}
    acc_len = {acc: int(ln) for acc, _, ln in cand}

    # client-side bucketing + per-bucket cap (whole-bucket for rare long classes)
    by_bucket: dict[str, list[str]] = {name: [] for name, _, _ in BUCKETS}
    for acc, _, ln in cand:
        by_bucket[bucket_of(int(ln))].append(acc)
    sample: list[str] = []
    for name in by_bucket:
        accs_b = by_bucket[name]
        rng.shuffle(accs_b)
        take = accs_b[:per_bucket_cap]
        log.info("  bucket %-9s available=%d -> taking %d", name, len(accs_b), len(take))
        sample.extend(take)
    rng.shuffle(sample)
    log.info("  length-balanced sample size: %d", len(sample))
    seq_ids = tuple(acc_to_seq[a] for a in sample)
    seq_to_acc = {acc_to_seq[a]: a for a in sample}
    sample_set = tuple(sample)

    log.info("pulling mean vectors (%s)", EMB_MEAN)
    cur.execute(
        """
        SELECT sequence_id, embedding::text
        FROM sequence_embedding
        WHERE embedding_config_id = %s AND sequence_id IN %s
        """,
        (EMB_MEAN, seq_ids),
    )
    mean: dict[str, np.ndarray] = {}
    for sid, vtext in cur:
        a = seq_to_acc.get(sid)
        if a is not None:
            mean[a] = _parse_vec(vtext)

    log.info("pulling learned codes (%s)", EMB_LEARNED)
    cur.execute(
        """
        SELECT sequence_id, embedding::text
        FROM sequence_embedding
        WHERE embedding_config_id = %s AND sequence_id IN %s
        """,
        (EMB_LEARNED, seq_ids),
    )
    learned: dict[str, np.ndarray] = {}
    for sid, vtext in cur:
        a = seq_to_acc.get(sid)
        if a is not None:
            learned[a] = _parse_vec(vtext)

    log.info("pulling per-chunk vectors (%s)", EMB_CHUNK)
    cur.execute(
        """
        SELECT sequence_id, chunk_index_s, embedding::text
        FROM sequence_embedding
        WHERE embedding_config_id = %s AND sequence_id IN %s
        ORDER BY sequence_id, chunk_index_s
        """,
        (EMB_CHUNK, seq_ids),
    )
    chunk_tmp: dict[str, list[tuple[int, np.ndarray]]] = {}
    for sid, ci, vtext in cur:
        a = seq_to_acc.get(sid)
        if a is not None:
            chunk_tmp.setdefault(a, []).append((ci, _parse_vec(vtext)))
    chunks: dict[str, np.ndarray] = {}
    for a, lst in chunk_tmp.items():
        lst.sort(key=lambda t: t[0])
        chunks[a] = np.vstack([v for _, v in lst])

    log.info("pulling v227 leaf annotations")
    cur.execute(
        """
        SELECT pga.protein_accession, gt.go_id
        FROM protein_go_annotation pga
        JOIN go_term gt ON gt.id = pga.go_term_id
        WHERE pga.annotation_set_id = %s AND pga.protein_accession IN %s
        """,
        (ANN_SET, sample_set),
    )
    leaves: dict[str, list[str]] = {}
    for a, go in cur:
        leaves.setdefault(a, []).append(go)

    cur.close()
    conn.close()

    accs = [
        a
        for a in sample
        if a in mean and a in learned and a in chunks and a in leaves
    ]
    log.info("  %d accessions with mean+chunk+learned+annotations", len(accs))
    return accs, acc_len, mean, chunks, learned, leaves


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------
def chunk_sdr_bundle(chunks: np.ndarray, k: int, d: int) -> np.ndarray:
    """Per-chunk k-WTA -> length-normalised top-k vote bundle (one uint8 SDR row)."""
    if chunks.shape[0] == 1:
        top = kwta_active_set(chunks[0], k)
    else:
        votes = np.zeros(d, dtype=np.int32)
        mass = np.zeros(d, dtype=np.float32)
        for chunk in chunks:
            act = kwta_active_set(chunk, k)
            votes[act] += 1
            mass[act] += np.abs(chunk[act])
        order = np.lexsort((mass, votes))[::-1]
        top = order[:k]
    out = np.zeros(d, dtype=np.uint8)
    out[top] = 1
    return out


def learned_active_bitset(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Binarise the learned k-WTA codes to their nonzero support (active set).

    The learned code is already sparse (k-WTA at fit time); its support size is the
    arm's natural density. Returns the (n, d) uint8 bitset and the median active count
    used as the Tanimoto density constant.
    """
    bits = (codes != 0.0).astype(np.uint8)
    k_active = int(np.median(bits.sum(axis=1))) or 1
    return bits, k_active


# ---------------------------------------------------------------------------
# Per-bucket pair sampling + Spearman
# ---------------------------------------------------------------------------
def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator):
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def go_targets(closures, ic, best_ic, pairs):
    return {
        "resnik": np.asarray(resnik_pairwise(closures, ic, pairs), dtype=np.float64),
        "lin": np.asarray(lin_pairwise(closures, ic, pairs, best_ic), dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Main readout
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    dag = GoDag.from_obo(Path(args.obo).expanduser())

    # ===================== DB arms (mean / chunk / learned) =====================
    accs, acc_len, mean, chunks, learned, leaves = load_db_sample(
        args.dsn, args.n_proteins, args.seed, args.per_bucket_cap
    )
    closures_all = [propagate(leaves[a], dag) for a in accs]
    keep = [i for i, c in enumerate(closures_all) if c]
    accs = [accs[i] for i in keep]
    closures_all = [closures_all[i] for i in keep]
    log.info("DB arms: %d proteins with a non-empty closure", len(accs))

    lengths = np.array([acc_len[a] for a in accs])
    log.info(
        "length dist: min=%d p25=%d median=%d p75=%d max=%d",
        int(lengths.min()), int(np.percentile(lengths, 25)),
        int(np.median(lengths)), int(np.percentile(lengths, 75)), int(lengths.max()),
    )

    # group indices by bucket
    buckets: dict[str, list[int]] = {name: [] for name, _, _ in BUCKETS}
    for i, ln in enumerate(lengths):
        buckets[bucket_of(int(ln))].append(i)

    d_mean = mean[accs[0]].shape[0]
    learned_bits_all, k_learned = learned_active_bitset(
        np.vstack([learned[a] for a in accs])
    )

    db_table: dict[str, dict] = {}  # bucket -> {arm_metric: rho, "N": n}
    for bname, idxs in buckets.items():
        n = len(idxs)
        entry: dict[str, float | int | None] = {"N": n}
        if n < args.min_bucket_n:
            log.warning("bucket %-9s N=%d below min (%d) -> arms skipped",
                        bname, n, args.min_bucket_n)
            db_table[bname] = entry
            continue

        sub_acc = [accs[i] for i in idxs]
        sub_clo = [closures_all[i] for i in idxs]
        ic = information_content(sub_clo, dag)
        best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in sub_clo]

        pairs = sample_pairs(n, args.n_pairs, rng)
        ii = np.array([p[0] for p in pairs])
        jj = np.array([p[1] for p in pairs])
        go = go_targets(sub_clo, ic, best_ic, pairs)
        log.info("bucket %-9s N=%d pairs=%d", bname, n, len(pairs))

        # dense-mean-cosine
        dense = np.vstack([mean[a] for a in sub_acc]).astype(np.float32)
        cos = cosine_dense(dense)[ii, jj]
        for m in ("resnik", "lin"):
            entry[f"dense-mean-cosine|{m}"] = float(spearmanr(cos, go[m])[0])

        # learned (fixed code; density from its own support)
        lbits = learned_bits_all[idxs]
        ltan = tanimoto_dense(lbits, k_learned)[ii, jj]
        for m in ("resnik", "lin"):
            entry[f"learned|{m}"] = float(spearmanr(ltan, go[m])[0])

        # mean-SDR and chunk-SDR, swept over k
        for k in args.kwta_k:
            msdr = kwta_binarise(dense, k)
            mtan = tanimoto_dense(msdr, k)[ii, jj]
            csdr = np.vstack(
                [chunk_sdr_bundle(chunks[a], k, d_mean) for a in sub_acc]
            )
            ctan = tanimoto_dense(csdr, k)[ii, jj]
            for m in ("resnik", "lin"):
                entry[f"mean-SDR-k{k}|{m}"] = float(spearmanr(mtan, go[m])[0])
                entry[f"chunk-SDR-k{k}|{m}"] = float(spearmanr(ctan, go[m])[0])

        db_table[bname] = entry

    # ===================== residue arm (own npy sample) =====================
    res_table = run_residue_arm(args, dag, rng)

    return {
        "n_db_proteins": len(accs),
        "db_table": db_table,
        "residue_table": res_table,
        "kwta_k": list(args.kwta_k),
        "buckets": [b[0] for b in BUCKETS],
        "k_learned": k_learned,
        "seed": args.seed,
    }


def run_residue_arm(args, dag, rng) -> dict:
    """Per-residue OR-bundle Tanimoto, over the ~5000 npy sample, bucketed by length.

    Length here is the residue count (npy rows) = the true sequence length.
    """
    import psycopg2

    files = [f for f in os.listdir(RESDIR) if f.endswith(".npy")]
    accs_all = [f[:-4] for f in files]
    log.info("residue arm: %d per-residue arrays on disk", len(accs_all))

    conn = psycopg2.connect(args.dsn)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pga.protein_accession, gt.go_id
        FROM protein_go_annotation pga
        JOIN go_term gt ON gt.id = pga.go_term_id
        WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)
        """,
        (ANN_SET, accs_all),
    )
    leaves: dict[str, list[str]] = {}
    for a, go in cur.fetchall():
        leaves.setdefault(a, []).append(go)
    cur.close()
    conn.close()

    acc, closures, len_list = [], [], []
    for a in accs_all:
        if a in leaves:
            c = propagate(leaves[a], dag)
            if c:
                M = np.load(os.path.join(RESDIR, f"{a}.npy"))
                acc.append(a)
                closures.append(c)
                len_list.append(M.shape[0])
    lengths = np.array(len_list)
    log.info("residue arm: %d usable proteins", len(acc))

    buckets: dict[str, list[int]] = {name: [] for name, _, _ in BUCKETS}
    for i, ln in enumerate(lengths):
        buckets[bucket_of(int(ln))].append(i)

    out: dict[str, dict] = {}
    for bname, idxs in buckets.items():
        n = len(idxs)
        entry: dict[str, float | int | None] = {"N": n}
        if n < args.min_bucket_n:
            log.warning("residue bucket %-9s N=%d below min (%d) -> skipped",
                        bname, n, args.min_bucket_n)
            out[bname] = entry
            continue

        sub_acc = [acc[i] for i in idxs]
        sub_clo = [closures[i] for i in idxs]
        ic = information_content(sub_clo, dag)
        best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in sub_clo]
        pairs = sample_pairs(n, args.n_pairs, rng)
        ii = np.array([p[0] for p in pairs])
        jj = np.array([p[1] for p in pairs])
        go = go_targets(sub_clo, ic, best_ic, pairs)
        log.info("residue bucket %-9s N=%d pairs=%d", bname, n, len(pairs))

        for k in args.kwta_k:
            sdr = build_residue_bundle(sub_acc, k)
            tan = tanimoto_dense(sdr, int(sdr.sum(1).mean()) or k)[ii, jj]
            for m in ("resnik", "lin"):
                entry[f"residue-SDR-k{k}|{m}"] = float(spearmanr(tan, go[m])[0])
        out[bname] = entry
    return out


def build_residue_bundle(sub_acc, k) -> np.ndarray:
    """For each protein: per-residue top-k (by |magnitude|), OR-bundle across residues."""
    d = 768
    out = np.zeros((len(sub_acc), d), dtype=np.uint8)
    for r, a in enumerate(sub_acc):
        M = np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32)
        kk = min(k, d - 1)
        idx = np.argpartition(-np.abs(M), kk, axis=1)[:, :kk]
        bits = np.zeros(M.shape, dtype=np.uint8)
        np.put_along_axis(bits, idx, 1, axis=1)
        out[r] = bits.any(0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Artifacts + MLflow
# ---------------------------------------------------------------------------
def _flatten_rows(result: dict) -> list[dict]:
    """One CSV row per (source, bucket): N + every arm|metric Spearman present."""
    rows = []
    for source, table in (("db", result["db_table"]), ("residue", result["residue_table"])):
        for bname in result["buckets"]:
            entry = table.get(bname, {})
            row = {"source": source, "bucket": bname, "N": entry.get("N", 0)}
            for kk, vv in entry.items():
                if kk != "N":
                    row[kk] = vv
            rows.append(row)
    return rows


def write_artifacts(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    rows = _flatten_rows(result)
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    csv = out_dir / "sdr_length_stratified.csv"
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(
            (f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            for c in cols
        ))
    csv.write_text("\n".join(lines) + "\n")
    paths.append(csv)

    js = out_dir / "sdr_length_stratified.json"
    js.write_text(json.dumps(result, indent=2))
    paths.append(js)
    return paths


def log_to_mlflow(args, result, artifacts) -> str | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return None
    try:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="sdr-length-stratified") as active:
            mlflow.log_params({
                "plm": "ankh-base",
                "emb_mean": EMB_MEAN,
                "emb_chunk": EMB_CHUNK,
                "emb_learned": EMB_LEARNED,
                "annotation_set": ANN_SET,
                "window": "v227-t0",
                "n_db_proteins": result["n_db_proteins"],
                "n_pairs_per_bucket": args.n_pairs,
                "kwta_k": ",".join(str(k) for k in args.kwta_k),
                "k_learned": result["k_learned"],
                "buckets": ",".join(result["buckets"]),
                "seed": args.seed,
            })
            for source, table in (("db", result["db_table"]),
                                  ("residue", result["residue_table"])):
                for bname, entry in table.items():
                    for kk, vv in entry.items():
                        if kk == "N":
                            mlflow.log_metric(f"N__{source}__{bname}", vv)
                        elif isinstance(vv, float):
                            key = f"{source}__{bname}__{kk}"
                            key = key.replace("|", "__").replace("-", "_")
                            mlflow.log_metric(key, vv)
            for p in artifacts:
                mlflow.log_artifact(str(p))
            run_id = active.info.run_id
        log.info("mlflow: run %s logged to %r", run_id, EXPERIMENT)
        return run_id
    except Exception as exc:  # pragma: no cover
        log.warning("mlflow logging failed (%s)", exc)
        return None


def _print_table(result: dict) -> None:
    metric_arms = []
    for source, table in (("db", result["db_table"]), ("residue", result["residue_table"])):
        for entry in table.values():
            for kk in entry:
                if kk != "N" and kk not in metric_arms:
                    metric_arms.append(kk)
    print("\n=== SDR length-stratified Spearman(arm, GO) ===")
    for source, table in (("db", result["db_table"]), ("residue", result["residue_table"])):
        print(f"\n[{source} sample]")
        header = f"{'bucket':<10} {'N':>6}  " + "  ".join(f"{a:<24}" for a in metric_arms
                                                          if any(a in table.get(b, {}) for b in result["buckets"]))
        print(header)
        for bname in result["buckets"]:
            entry = table.get(bname, {})
            cells = []
            for a in metric_arms:
                if any(a in table.get(b, {}) for b in result["buckets"]):
                    v = entry.get(a)
                    cells.append(f"{v:<24.4f}" if isinstance(v, float) else f"{'-':<24}")
            print(f"{bname:<10} {entry.get('N', 0):>6}  " + "  ".join(cells))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    p.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    p.add_argument("--n-proteins", type=int, default=8000,
                   help="(retained for compatibility; per-bucket-cap drives sizing)")
    p.add_argument("--per-bucket-cap", type=int, default=2000,
                   help="max DB proteins per length bucket (rare long buckets taken whole)")
    p.add_argument("--n-pairs", type=int, default=20_000, help="protein pairs per bucket")
    p.add_argument("--kwta-k", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--min-bucket-n", type=int, default=50,
                   help="buckets below this protein count are skipped + flagged")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "sdr_length_out"
    artifacts = write_artifacts(result, out_dir)
    run_id = None if args.no_mlflow else log_to_mlflow(args, result, artifacts)
    _print_table(result)
    if run_id:
        print(f"\nMLflow run: {run_id}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
