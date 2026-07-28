#!/usr/bin/env python
"""SDR-chunk readout: the FAIR re-test of sparse.pdf section 2 (sparsify-then-BUNDLE).

SDR-A tested "pool-then-sparsify": k-WTA on the already-mean-pooled ProtT5 vector,
and lost to dense cosine (Spearman vs GO-Resnik: cosine 0.315 vs Tanimoto 0.255).
sparse.pdf section 2 says the SDR-NATIVE route is the OTHER order: sparsify EACH
chunk, then BUNDLE the active sets (thresholded superposition / OR, length-normalised
so density does not saturate with chunk count). SDR-A never tested that order because
it only had mean-pooled vectors. We now have per-chunk embeddings (ankh-base chunked,
EmbeddingConfig 6542db1e..., 768-dim, one SequenceEmbedding row PER CHUNK), so we can.

Pipeline (frozen v227 reference pool, t0-only, annotation set c905dffa..., leakage-clean):

  1. Pull the per-chunk ankh-base vectors grouped by sequence (ordered by chunk_index_s)
     and the v227 GO leaf annotations for the SAME proteins.
  2. Sample N proteins carrying both >=1 chunk and >=1 annotation that maps into the DAG.
  3. For each protein build THREE representations:
       - DENSE  : mean of the per-chunk vectors -> cosine                 (baseline)
       - CHUNK-SDR (the new arm): k-WTA each chunk -> BUNDLE (count active
         coords across chunks, length-normalise, keep the top-k bundled
         coords) -> a compact protein SDR -> Tanimoto. Sweep k in {32,64,128}.
       - SDR-A control (optional): k-WTA on the MEAN vector -> Tanimoto    (contrast)
  4. Build TPR closures + Resnik IC over the sampled propagated corpus.
  5. Sample M protein pairs; Spearman-correlate each representation against GO
     semantic similarity (Resnik AND Lin).

GATE: does CHUNK-SDR (sparsify-then-bundle) close or BEAT the dense cosine gap that
SDR-A's pool-then-sparsify lost?

Logs to MLflow experiment ``sdr-chunk-correlation``.

Run inside the lab venv with MLflow live::

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    .venv/bin/python scripts/run_sdr_chunk_correlation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import psycopg2
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
    format="%(asctime)s [sdr-chunk] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT = "sdr-chunk-correlation"

EMB_CONFIG = "6542db1e-202a-4769-b933-2e0f85aa81e6"  # ankh-base chunked cs=512, 768-dim
ANN_SET = "2394b9a1-21df-4c2b-89af-3da084318fab"  # GOA v227, t0


# ---------------------------------------------------------------------------
# DB loading (READ-ONLY)
# ---------------------------------------------------------------------------
def _parse_halfvec(text: str) -> np.ndarray:
    """Parse a pgvector halfvec text literal ``[a,b,c]`` into a float32 array."""
    return np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float32)


def load_chunked(
    dsn: str, n_target: int, seed: int, min_chunks: int = 1
) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[str]]]:
    """Load per-chunk vectors (grouped per accession) + v227 GO leaf annotations.

    Returns (accessions, acc -> (n_chunks, d) chunk matrix, acc -> [GO:xxxx]). To keep
    RAM modest we first pick the candidate accession set (annotated + embedded), sample
    ``n_target`` of them with the SAME rng/seed as the readout, and only then pull their
    chunk vectors and annotations.
    """
    rng = np.random.default_rng(seed)
    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True)
    cur = conn.cursor()

    # The full DISTINCT(annotated INNER JOIN embedded) aggregate is a 5.8M-row scan and
    # starves under live-box I/O contention. We exploit the verified invariant that EVERY
    # v227-annotated protein also carries this embedding config (full overlap), so sampling
    # candidates from the annotation set alone is sufficient. We draw a bounded, pseudo-random
    # slice via a hash of the accession (deterministic in `seed`) over the indexed
    # annotation_set_id, then resolve sequence_id and pull chunks by sequence_id.
    # oversample: drop single-chunk-less / empty-closure later. Multi-chunk proteins are
    # only ~25% of the pool (ankh-base cs=512, most proteins <=512 aa), so when we require
    # >=2 chunks we must draw ~5x more candidates to land n_target survivors.
    over = 6.0 if min_chunks >= 2 else 1.4
    n_pool = int(n_target * over)
    # ~556k v227-annotated accessions; pick a hash bucket whose 1/bucket share ~= n_pool.
    bucket = max(2, int(round(556_000 / max(n_pool, 1))))
    salt = int(seed)
    log.info(
        "sampling ~%d candidate accessions from the annotation set (hash bucket 1/%d)",
        n_pool, bucket,
    )
    cur.execute(
        """
        SELECT p.accession, p.sequence_id
        FROM (
            SELECT DISTINCT protein_accession AS acc
            FROM protein_go_annotation
            WHERE annotation_set_id = %(ann)s
              AND (hashtextextended(protein_accession, %(salt)s) %% %(bucket)s) = 0
        ) s
        JOIN protein p ON p.accession = s.acc
        WHERE p.sequence_id IS NOT NULL
        """,
        {"ann": ANN_SET, "salt": salt, "bucket": bucket},
    )
    cand_rows = cur.fetchall()
    log.info("  %d candidate accessions with a sequence_id", len(cand_rows))
    acc_to_seq = {acc: sid for acc, sid in cand_rows}
    candidates = list(acc_to_seq)
    rng.shuffle(candidates)
    sample = candidates[:n_pool]
    sample_set = tuple(sample)
    seq_ids = tuple(acc_to_seq[a] for a in sample)
    seq_to_acc = {acc_to_seq[a]: a for a in sample}

    log.info("pulling per-chunk vectors for %d sampled sequences", len(seq_ids))
    cur.execute(
        """
        SELECT se.sequence_id, se.chunk_index_s, se.embedding::text
        FROM sequence_embedding se
        WHERE se.embedding_config_id = %s
          AND se.sequence_id IN %s
        ORDER BY se.sequence_id, se.chunk_index_s
        """,
        (EMB_CONFIG, seq_ids),
    )
    chunks: dict[str, list[tuple[int, np.ndarray]]] = {}
    for sid, ci, vtext in cur:
        acc = seq_to_acc.get(sid)
        if acc is not None:
            chunks.setdefault(acc, []).append((ci, _parse_halfvec(vtext)))

    acc_chunks: dict[str, np.ndarray] = {}
    for acc, lst in chunks.items():
        lst.sort(key=lambda t: t[0])
        mat = np.vstack([v for _, v in lst])
        if mat.shape[0] >= min_chunks:
            acc_chunks[acc] = mat

    log.info("pulling v227 annotations for sampled accessions")
    cur.execute(
        """
        SELECT pga.protein_accession, gt.go_id
        FROM protein_go_annotation pga
        JOIN go_term gt ON gt.id = pga.go_term_id
        WHERE pga.annotation_set_id = %s
          AND pga.protein_accession IN %s
        """,
        (ANN_SET, sample_set),
    )
    leaves: dict[str, list[str]] = {}
    for acc, go in cur:
        leaves.setdefault(acc, []).append(go)

    cur.close()
    conn.close()

    accessions = [a for a in sample if a in acc_chunks and a in leaves]
    log.info("  %d accessions with chunks AND annotations", len(accessions))
    return accessions, acc_chunks, leaves


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------
def dense_mean(acc_chunks: dict[str, np.ndarray], accessions: list[str]) -> np.ndarray:
    """Mean-pool the per-chunk vectors per protein -> (n, d) dense matrix (baseline)."""
    return np.vstack([acc_chunks[a].mean(axis=0) for a in accessions]).astype(np.float32)


def chunk_sdr_bundle(
    acc_chunks: dict[str, np.ndarray], accessions: list[str], k: int, d: int
) -> np.ndarray:
    """Sparsify-then-BUNDLE: per-chunk k-WTA, then a length-normalised top-k bundle.

    For each protein:
      * k-WTA each chunk vector -> a per-chunk active set of size k.
      * BUNDLE by superposition: count how often each coordinate is active across the
        protein's chunks (a thresholded-OR vote in counting space).
      * Length-normalise: instead of an absolute count threshold (which would saturate
        density as chunk count grows), keep the top-k coordinates by vote (ties broken
        by total k-WTA magnitude mass, then index). This yields a FIXED-k protein SDR
        regardless of chunk count, so density is constant and Tanimoto's i/(2k-i)
        formula (which assumes exactly k active bits per row) holds exactly.

    Returns a (n, d) uint8 bitset with exactly k active bits per row.
    """
    out = np.zeros((len(accessions), d), dtype=np.uint8)
    for r, acc in enumerate(accessions):
        mat = acc_chunks[acc]
        votes = np.zeros(d, dtype=np.int32)
        mass = np.zeros(d, dtype=np.float32)  # tie-break: summed magnitude of active coords
        for chunk in mat:
            act = kwta_active_set(chunk, k)
            votes[act] += 1
            mass[act] += np.abs(chunk[act])
        if mat.shape[0] == 1:
            # single chunk: the bundle is just that chunk's active set
            top = kwta_active_set(mat[0], k)
        else:
            # rank by (vote count, magnitude mass); keep top-k for a fixed-density SDR
            order = np.lexsort((mass, votes))[::-1]  # primary votes, secondary mass
            top = order[:k]
        out[r, top] = 1
    return out


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    dsn = args.dsn

    min_chunks = 2 if args.multi_chunk_only else 1
    accessions, acc_chunks, leaves = load_chunked(
        dsn, args.n_proteins, args.seed, min_chunks=min_chunks
    )
    if args.multi_chunk_only:
        log.info("MULTI-CHUNK-ONLY: bundling is only non-trivial when n_chunks>=2")

    log.info("parsing GO DAG from %s", args.obo)
    dag = GoDag.from_obo(Path(args.obo).expanduser())

    log.info("building TPR closures")
    closures = [propagate(leaves[a], dag) for a in accessions]
    keep = [i for i, c in enumerate(closures) if c]
    accessions = [accessions[i] for i in keep]
    closures = [closures[i] for i in keep]
    accessions = accessions[: args.n_proteins]
    closures = closures[: args.n_proteins]
    log.info("  %d proteins with a non-empty closure", len(accessions))

    n_chunks = np.array([acc_chunks[a].shape[0] for a in accessions])
    log.info(
        "chunk counts: mean=%.2f median=%d max=%d (single-chunk: %.1f%%)",
        n_chunks.mean(), int(np.median(n_chunks)), int(n_chunks.max()),
        100.0 * (n_chunks == 1).mean(),
    )

    d = next(iter(acc_chunks.values())).shape[1]
    log.info("embedding dim d=%d", d)

    log.info("information content over the sampled propagated corpus")
    ic = information_content(closures, dag)
    best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]

    # sample distinct pairs
    n = len(accessions)
    n_pairs = min(args.n_pairs, n * (n - 1) // 2)
    log.info("sampling %d protein pairs", n_pairs)
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
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])

    # representations
    log.info("building DENSE (mean-pool) representation")
    dense = dense_mean(acc_chunks, accessions)

    results: dict[str, dict] = {}
    for metric in ("resnik", "lin"):
        log.info("GO semantic similarity: %s", metric)
        if metric == "resnik":
            go_sim = resnik_pairwise(closures, ic, pairs)
        else:
            go_sim = lin_pairwise(closures, ic, pairs, best_ic)

        cos = cosine_dense(dense)[ii, jj]
        rho_cos, p_cos = spearmanr(cos, go_sim)
        log.info("  Spearman(cosine, GO-%s) = %.4f (p=%.2e)", metric, rho_cos, p_cos)

        chunk_per_k: dict[int, dict[str, float]] = {}
        ctrl_per_k: dict[int, dict[str, float]] = {}
        for k in args.kwta_k:
            sdr_chunk = chunk_sdr_bundle(acc_chunks, accessions, k, d)
            tan = tanimoto_dense(sdr_chunk, k)[ii, jj]
            rho_c, p_c = spearmanr(tan, go_sim)
            chunk_per_k[k] = {"spearman": float(rho_c), "p_value": float(p_c)}
            log.info("  Spearman(chunkSDR k=%d, GO-%s) = %.4f", k, metric, rho_c)

            if args.with_control:
                sdr_a = kwta_binarise(dense, k)
                tan_a = tanimoto_dense(sdr_a, k)[ii, jj]
                rho_a, p_a = spearmanr(tan_a, go_sim)
                ctrl_per_k[k] = {"spearman": float(rho_a), "p_value": float(p_a)}
                log.info("  Spearman(SDR-A ctrl k=%d, GO-%s) = %.4f", k, metric, rho_a)

        best_k = max(chunk_per_k, key=lambda kk: chunk_per_k[kk]["spearman"])
        best_chunk = chunk_per_k[best_k]["spearman"]
        gap = best_chunk - rho_cos
        verdict = "BEATS" if gap >= 0 else ("CLOSES" if gap >= -0.02 else "STILL-BELOW")
        log.info(
            "  GATE[%s]: best chunkSDR rho=%.4f (k=%d) vs cosine %.4f -> gap %+.4f -> %s",
            metric, best_chunk, best_k, rho_cos, gap, verdict,
        )

        results[metric] = {
            "spearman_cosine_go": float(rho_cos),
            "p_cosine": float(p_cos),
            "chunk_per_k": chunk_per_k,
            "control_per_k": ctrl_per_k,
            "best_k": int(best_k),
            "best_chunksdr_rho": float(best_chunk),
            "gap_vs_cosine": float(gap),
            "verdict": verdict,
        }

    return {
        "n_proteins": len(accessions),
        "n_pairs": n_pairs,
        "embedding_dim": d,
        "embedding_config": EMB_CONFIG,
        "annotation_set": ANN_SET,
        "kwta_k": list(args.kwta_k),
        "seed": args.seed,
        "chunk_count_mean": float(n_chunks.mean()),
        "chunk_count_median": int(np.median(n_chunks)),
        "single_chunk_frac": float((n_chunks == 1).mean()),
        "per_metric": results,
    }


# ---------------------------------------------------------------------------
# Artifacts + MLflow
# ---------------------------------------------------------------------------
def write_artifacts(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    csv = out_dir / "sdr_chunk_summary.csv"
    lines = ["semantic_metric,arm,k,spearman_vs_go"]
    for metric, m in result["per_metric"].items():
        lines.append(f"{metric},dense_cosine,,{m['spearman_cosine_go']:.6f}")
        for k, v in m["chunk_per_k"].items():
            lines.append(f"{metric},chunk_sdr_tanimoto,{k},{v['spearman']:.6f}")
        for k, v in m["control_per_k"].items():
            lines.append(f"{metric},sdr_a_control_tanimoto,{k},{v['spearman']:.6f}")
    csv.write_text("\n".join(lines) + "\n")
    paths.append(csv)

    js = out_dir / "sdr_chunk_result.json"
    js.write_text(json.dumps(result, indent=2))
    paths.append(js)
    return paths


def log_to_mlflow(args: argparse.Namespace, result: dict, artifacts: list[Path]) -> str | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return None
    try:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        run_name = "sdr-chunk-bundle-multichunk" if args.multi_chunk_only else "sdr-chunk-bundle"
        with mlflow.start_run(run_name=run_name) as active:
            mlflow.set_tag("multi_chunk_only", str(args.multi_chunk_only))
            mlflow.log_params({
                "plm": "ankh-base",
                "embedding_config": EMB_CONFIG,
                "annotation_set": ANN_SET,
                "window": "v230-t1",
                "n_proteins": result["n_proteins"],
                "n_pairs": result["n_pairs"],
                "kwta_k": ",".join(str(k) for k in args.kwta_k),
                "seed": args.seed,
                "embedding_dim": result["embedding_dim"],
                "bundle": "sparsify-then-bundle (per-chunk k-WTA -> top-k vote)",
                "chunk_count_mean": round(result["chunk_count_mean"], 3),
                "single_chunk_frac": round(result["single_chunk_frac"], 4),
            })
            for metric, m in result["per_metric"].items():
                mlflow.log_metric(f"spearman_cosine_go_{metric}", m["spearman_cosine_go"])
                for k, v in m["chunk_per_k"].items():
                    mlflow.log_metric(f"spearman_chunksdr_go_{metric}_k{k}", v["spearman"])
                for k, v in m["control_per_k"].items():
                    mlflow.log_metric(f"spearman_sdra_ctrl_go_{metric}_k{k}", v["spearman"])
                mlflow.log_metric(f"best_chunksdr_rho_{metric}", m["best_chunksdr_rho"])
                mlflow.log_metric(f"gap_vs_cosine_{metric}", m["gap_vs_cosine"])
                mlflow.set_tag(f"verdict_{metric}", m["verdict"])
            mlflow.set_tag("slice", "SDR-chunk")
            for p in artifacts:
                mlflow.log_artifact(str(p))
            run_id = active.info.run_id
        log.info("mlflow: run %s logged to experiment %r", run_id, EXPERIMENT)
        return run_id
    except Exception as exc:  # pragma: no cover
        log.warning("mlflow logging failed (%s)", exc)
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    p.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    p.add_argument("--n-proteins", type=int, default=5000)
    p.add_argument("--n-pairs", type=int, default=200_000)
    p.add_argument("--kwta-k", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--with-control", action="store_true", default=True,
                   help="also run the SDR-A control (k-WTA on the mean)")
    p.add_argument("--no-control", dest="with_control", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi-chunk-only", action="store_true",
                   help="restrict to proteins with >=2 chunks (where bundling is non-trivial)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "sdr_chunk_out"
    artifacts = write_artifacts(result, out_dir)
    run_id = None if args.no_mlflow else log_to_mlflow(args, result, artifacts)

    print("\n=== SDR-chunk readout (sparsify-then-BUNDLE) ===")
    print(f"proteins={result['n_proteins']}  pairs={result['n_pairs']}  "
          f"dim={result['embedding_dim']}  chunks/prot mean={result['chunk_count_mean']:.2f} "
          f"single-chunk={100*result['single_chunk_frac']:.1f}%")
    for metric, m in result["per_metric"].items():
        print(f"\n[{metric.upper()}]")
        print(f"  dense cosine            Spearman vs GO = {m['spearman_cosine_go']:.4f}")
        for k, v in m["chunk_per_k"].items():
            print(f"  chunk-SDR Tanimoto k={k:<4} Spearman vs GO = {v['spearman']:.4f}")
        for k, v in m["control_per_k"].items():
            print(f"  SDR-A ctrl Tanimoto k={k:<4} Spearman vs GO = {v['spearman']:.4f}")
        print(f"  GATE: best chunk-SDR rho={m['best_chunksdr_rho']:.4f} (k={m['best_k']}) "
              f"gap vs cosine {m['gap_vs_cosine']:+.4f} -> {m['verdict']}")
    if run_id:
        print(f"\nMLflow run: {run_id}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
