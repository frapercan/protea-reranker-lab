#!/usr/bin/env python
"""SDR-A readout-1: overlap-vs-GO-semantic correlation (T-CIENCIA, sparse.pdf App. A.2).

The cheapest decisive test of the sparse.pdf hypothesis on PROTEA's own benchmark:
does an interpretable set-overlap in a SPARSE space (k-WTA + Tanimoto) track GO
semantic similarity AT LEAST AS WELL as a distance in the DENSE space (cosine), over
the same protein pairs, on the SELECT window, leakage-clean?

Pipeline (all on the frozen v227 = GOA-227 reference pool, t0-only, no t1 leakage):

  1. Load the cached ProtT5 embeddings (1024-dim, raw) + the v227 GO annotations.
  2. Sample N reference proteins that carry both an embedding and >=1 annotation.
  3. Build each protein's True-Path-Rule GO closure over the t0 ontology snapshot;
     derive Resnik information content from the propagated corpus.
  4. Sample M protein pairs; for each pair compute
       - dense cosine                       (the BASELINE arm)
       - SDR Tanimoto for each k in --kwta-k (the SPARSE arm, k-WTA binarisation)
       - GO semantic similarity (Resnik or Lin over the DAG)  (the ground truth)
  5. Spearman-correlate (cosine, GO) and (Tanimoto_k, GO).

GATE: if any Tanimoto_k correlates with GO semantics at least as well as cosine
(within a small tolerance), the hypothesis HOLDS -> proceed to a full SDR-A k-NN arm;
otherwise the result is recorded cleanly as NEGATIVE.

Everything is logged to MLflow under experiment ``sdr-a-correlation`` (params, the
per-k Spearman metrics, the summary table, and the scatter plot).

Run inside the lab venv with MLflow live::

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    poetry run python scripts/run_sdr_a_correlation.py \\
        --bundle ~/Thesis2/storage/protea-frozen-v227-2025-09-04 \\
        --obo ~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from protea_reranker_lab.sdr import (
    GoDag,
    cosine_dense,
    information_content,
    kwta_binarise,
    lin_pairwise,
    propagate,
    resnik_pairwise,
    tanimoto_dense,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sdr-a] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT = "sdr-a-correlation"


# ---------------------------------------------------------------------------
# Data loading (the SELECT-window t0 reference pool)
# ---------------------------------------------------------------------------
def load_bundle(bundle: Path) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """Load (accessions, embedding matrix, accession -> [GO id]) from the frozen pool.

    The bundle is the leakage-clean v227 reference: ProtT5 embeddings + the t0 GO
    annotations + the go_term_id -> GO:xxxx metadata bridge.
    """
    log.info("loading embeddings from %s", bundle / "reference_embeddings.parquet")
    emb_tbl = pq.read_table(bundle / "reference_embeddings.parquet")
    accessions = emb_tbl.column("accession").to_pylist()
    emb = np.asarray(emb_tbl.column("embedding").to_pylist(), dtype=np.float32)
    log.info("  %d embeddings, dim=%d", emb.shape[0], emb.shape[1])

    log.info("loading go_term_id -> GO:xxxx metadata")
    meta = pq.read_table(bundle / "go_term_metadata.parquet")
    id_to_go = dict(zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist()))

    log.info("loading v227 annotations")
    ann = pq.read_table(
        bundle / "reference_annotations.parquet", columns=["accession", "go_term_id"]
    )
    ann_acc = ann.column("accession").to_pylist()
    ann_term = ann.column("go_term_id").to_pylist()
    leaves: dict[str, list[str]] = {}
    for acc, tid in zip(ann_acc, ann_term):
        go = id_to_go.get(tid)
        if go is not None:
            leaves.setdefault(acc, []).append(go)
    log.info("  %d annotated proteins", len(leaves))
    return accessions, emb, leaves


# ---------------------------------------------------------------------------
# Sampling + the correlation readout
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    bundle = Path(args.bundle).expanduser()
    rng = np.random.default_rng(args.seed)

    accessions, emb, leaves = load_bundle(bundle)
    acc_to_row = {a: i for i, a in enumerate(accessions)}

    log.info("parsing GO DAG from %s", args.obo)
    dag = GoDag.from_obo(Path(args.obo).expanduser())

    # candidate proteins: have an embedding AND >=1 annotation that maps into the DAG
    candidates = [a for a in leaves if a in acc_to_row]
    log.info("candidate proteins (embedding + annotation): %d", len(candidates))
    rng.shuffle(candidates)
    sample = candidates[: args.n_proteins]

    # build closures + per-aspect-aware information content over the PROPAGATED corpus
    log.info("building TPR closures for %d sampled proteins", len(sample))
    closures = [propagate(leaves[a], dag) for a in sample]
    keep = [i for i, c in enumerate(closures) if c]  # drop proteins whose labels were all obsolete
    sample = [sample[i] for i in keep]
    closures = [closures[i] for i in keep]
    log.info("  %d proteins with a non-empty closure", len(sample))

    rows = np.array([acc_to_row[a] for a in sample], dtype=np.int64)
    sub_emb = emb[rows]

    log.info("computing information content over the sampled propagated corpus")
    ic = information_content(closures, dag)
    best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]

    # sample pairs of DISTINCT proteins
    n = len(sample)
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

    # GO semantic similarity (the ground truth)
    log.info("computing GO semantic similarity (%s)", args.semantic_metric)
    if args.semantic_metric == "resnik":
        go_sim = resnik_pairwise(closures, ic, pairs)
    else:
        go_sim = lin_pairwise(closures, ic, pairs, best_ic)

    # dense cosine (BASELINE) for the same pairs
    log.info("computing dense cosine for the sampled pairs")
    cos_full = cosine_dense(sub_emb)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    cos = cos_full[ii, jj]

    rho_cos, p_cos = spearmanr(cos, go_sim)
    log.info("Spearman(cosine, GO-%s) = %.4f (p=%.2e)", args.semantic_metric, rho_cos, p_cos)

    # SDR Tanimoto (SPARSE) for each k
    per_k: dict[int, dict[str, float]] = {}
    for k in args.kwta_k:
        log.info("k-WTA binarise + Tanimoto for k=%d", k)
        sdr = kwta_binarise(sub_emb, k)
        tan_full = tanimoto_dense(sdr, k)
        tan = tan_full[ii, jj]
        rho_tan, p_tan = spearmanr(tan, go_sim)
        per_k[k] = {"spearman_tanimoto_go": float(rho_tan), "p_value": float(p_tan)}
        log.info("  Spearman(tanimoto_k=%d, GO) = %.4f (p=%.2e)", k, rho_tan, p_tan)

    # GATE verdict
    best_k = max(per_k, key=lambda kk: per_k[kk]["spearman_tanimoto_go"])
    best_tan = per_k[best_k]["spearman_tanimoto_go"]
    holds = best_tan >= (rho_cos - args.tolerance)
    verdict = "HOLD" if holds else "NEGATIVE"
    log.info(
        "GATE: best Tanimoto rho=%.4f (k=%d) vs cosine rho=%.4f (tol=%.3f) -> %s",
        best_tan, best_k, rho_cos, args.tolerance, verdict,
    )

    return {
        "n_candidates": len(candidates),
        "n_proteins": len(sample),
        "n_pairs": n_pairs,
        "semantic_metric": args.semantic_metric,
        "spearman_cosine_go": float(rho_cos),
        "p_cosine": float(p_cos),
        "per_k": per_k,
        "best_k": int(best_k),
        "best_tanimoto_rho": float(best_tan),
        "tolerance": args.tolerance,
        "verdict": verdict,
        "_arrays": {"cos": cos, "go_sim": go_sim, "pairs": pairs, "sub_emb": sub_emb,
                    "ii": ii, "jj": jj, "kwta_k": list(args.kwta_k)},
    }


# ---------------------------------------------------------------------------
# Artifacts + MLflow
# ---------------------------------------------------------------------------
def write_artifacts(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = result.pop("_arrays")
    paths: list[Path] = []

    # summary table (csv)
    csv = out_dir / "sdr_a_summary.csv"
    lines = ["arm,k,spearman_vs_go,p_value"]
    lines.append(f"dense_cosine,,{result['spearman_cosine_go']:.6f},{result['p_cosine']:.3e}")
    for k, m in result["per_k"].items():
        lines.append(f"sdr_tanimoto,{k},{m['spearman_tanimoto_go']:.6f},{m['p_value']:.3e}")
    csv.write_text("\n".join(lines) + "\n")
    paths.append(csv)

    # json summary
    js = out_dir / "sdr_a_result.json"
    js.write_text(json.dumps(result, indent=2))
    paths.append(js)

    # scatter: cosine + best-k Tanimoto vs GO semantic sim
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sub = min(4000, len(arr["go_sim"]))
        sel = np.random.default_rng(0).choice(len(arr["go_sim"]), sub, replace=False)
        go = arr["go_sim"][sel]

        axes[0].scatter(arr["cos"][sel], go, s=3, alpha=0.25)
        axes[0].set_title(f"dense cosine vs GO sim\nSpearman={result['spearman_cosine_go']:.3f}")
        axes[0].set_xlabel("cosine similarity")
        axes[0].set_ylabel(f"GO {result['semantic_metric']} sim")

        bk = result["best_k"]
        sdr = kwta_binarise(arr["sub_emb"], bk)
        tan = tanimoto_dense(sdr, bk)[arr["ii"], arr["jj"]]
        axes[1].scatter(tan[sel], go, s=3, alpha=0.25, color="C1")
        axes[1].set_title(
            f"SDR Tanimoto (k={bk}) vs GO sim\nSpearman={result['best_tanimoto_rho']:.3f}"
        )
        axes[1].set_xlabel("Tanimoto similarity")
        axes[1].set_ylabel(f"GO {result['semantic_metric']} sim")

        fig.suptitle(f"SDR-A readout-1 (verdict: {result['verdict']})")
        fig.tight_layout()
        png = out_dir / "sdr_a_scatter.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        paths.append(png)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        log.warning("scatter plot skipped (%s)", exc)

    return paths


def log_to_mlflow(args: argparse.Namespace, result: dict, artifacts: list[Path]) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return
    try:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"sdr-a-{args.semantic_metric}"):
            mlflow.log_params({
                "plm": "prostt5",
                "K": 5,
                "window": "SELECT",
                "n_proteins": result["n_proteins"],
                "n_pairs": result["n_pairs"],
                "kwta_k": ",".join(str(k) for k in args.kwta_k),
                "semantic_metric": args.semantic_metric,
                "seed": args.seed,
                "embedding_dim": 1024,
                "reference_cutoff": "v227",
            })
            mlflow.log_metric("spearman_cosine_go", result["spearman_cosine_go"])
            for k, m in result["per_k"].items():
                mlflow.log_metric("spearman_tanimoto_go", m["spearman_tanimoto_go"], step=k)
                mlflow.log_metric(f"spearman_tanimoto_go_k{k}", m["spearman_tanimoto_go"])
            mlflow.log_metric("best_tanimoto_rho", result["best_tanimoto_rho"])
            mlflow.set_tag("verdict", result["verdict"])
            mlflow.set_tag("slice", "SDR-A")
            for p in artifacts:
                mlflow.log_artifact(str(p))
        log.info("mlflow: run logged to experiment %r", EXPERIMENT)
    except Exception as exc:  # pragma: no cover - best-effort
        log.warning("mlflow logging failed (%s)", exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="~/Thesis2/storage/protea-frozen-v227-2025-09-04",
                   help="frozen v227 reference bundle (embeddings + annotations + metadata)")
    p.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo",
                   help="t0 GO ontology snapshot (go-basic.obo matching the cutoff)")
    p.add_argument("--n-proteins", type=int, default=5000,
                   help="number of reference proteins to sample")
    p.add_argument("--n-pairs", type=int, default=200_000,
                   help="number of protein pairs to correlate")
    p.add_argument("--kwta-k", type=int, nargs="+", default=[32, 64, 128],
                   help="k values for the k-WTA active set (sparse.pdf sweep)")
    p.add_argument("--semantic-metric", choices=["resnik", "lin"], default="resnik")
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="how much below cosine the best Tanimoto may fall and still HOLD")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="artifact output dir (default: ./sdr_a_out)")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "sdr_a_out"
    artifacts = write_artifacts(result, out_dir)
    if not args.no_mlflow:
        log_to_mlflow(args, result, artifacts)

    print("\n=== SDR-A readout-1 summary ===")
    print(f"proteins={result['n_proteins']}  pairs={result['n_pairs']}  "
          f"semantic={result['semantic_metric']}")
    print(f"dense cosine     Spearman vs GO = {result['spearman_cosine_go']:.4f}")
    for k, m in result["per_k"].items():
        print(f"SDR Tanimoto k={k:<4} Spearman vs GO = {m['spearman_tanimoto_go']:.4f}")
    print(f"GATE verdict: {result['verdict']} "
          f"(best Tanimoto rho={result['best_tanimoto_rho']:.4f} at k={result['best_k']}, "
          f"cosine rho={result['spearman_cosine_go']:.4f})")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
