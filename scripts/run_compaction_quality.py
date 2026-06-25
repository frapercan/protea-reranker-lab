"""Compaction-quality study: which compaction of a protein's per-residue signal best
PRESERVES that signal, per storage byte, across ALL protein lengths?

Backbone ankh-base (768d). Per-residue tokens are computed TRANSIENTLY (one forward per
sampled protein on GPU) and never persisted -- the dense per-residue corpus (~283 GB) is
infeasible to store, which is exactly why the compaction is the science. The gold
"full-signal" similarity is the late-interaction max-sim over residue sets (Sinkhorn-OT
as a control). Candidate compactions (mean, max-pool, PCA-of-mean, SDR-union,
multi-vector) are swept over a storage budget; preservation is Spearman(compact-sim,
gold-sim), stratified by length bucket. Everything is logged to MLflow as a parent run
plus one nested run per (compaction x budget) so the Pareto preservation-vs-bytes curve
charts live.

Read-only DB. Single heavy job. One PLM. Lazy torch import.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/home/frapercan/Thesis2/repositories/protea-backends/src")

from protea_reranker_lab.compaction_quality import (  # noqa: E402
    BUCKET_NAMES,
    cosine_pairwise_fn,
    length_bucket,
    max_pool,
    mean_of_reps,
    mean_pool,
    multivector_codes,
    multivector_pairwise_fn,
    normweighted_mean_of_reps,
    pca_fit_transform,
    representative_residues,
    sdr_union_codes,
    sdr_union_of_reps,
    sinkhorn_ot_sim,
    tanimoto_pairwise_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [compaction] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compaction")

ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0 reference pool
MODEL = "ElnaggarLab/ankh-base"
DSN = "host=localhost dbname=protea user=protea password=protea"
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT = "compaction-quality"


# ---------------------------------------------------------------------------
# Stratified sampling from the read-only DB
# ---------------------------------------------------------------------------
def sample_stratified(n_per_bucket: int, seed: int) -> list[tuple[str, int, str]]:
    """Sample ``n_per_bucket`` v227-annotated proteins per length bucket (acc, len, seq)."""
    import psycopg2

    rng = np.random.default_rng(seed)
    bounds = [(1, 318), (319, 969), (970, 1959), (1960, 100000)]
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    out: list[tuple[str, int, str]] = []
    for (lo, hi), name in zip(bounds, BUCKET_NAMES, strict=True):
        cur.execute(
            """
            SELECT p.accession, p.length, s.sequence
            FROM (SELECT DISTINCT protein_accession FROM protein_go_annotation
                  WHERE annotation_set_id = %s) a
            JOIN protein p ON p.accession = a.protein_accession
            JOIN sequence s ON s.id = p.sequence_id
            WHERE p.length BETWEEN %s AND %s AND s.sequence IS NOT NULL
            """,
            (ANN_SET, lo, hi),
        )
        rows = cur.fetchall()
        if len(rows) > n_per_bucket:
            idx = rng.choice(len(rows), size=n_per_bucket, replace=False)
            rows = [rows[i] for i in idx]
        log.info("bucket %-10s candidates sampled: %d", name, len(rows))
        out.extend((acc, int(length), seq) for acc, length, seq in rows)
    cur.close()
    conn.close()
    return out


# ---------------------------------------------------------------------------
# Transient ankh-base per-residue forward (GPU, never persisted)
# ---------------------------------------------------------------------------
def forward_residues(
    proteins: list[tuple[str, int, str]],
) -> tuple[list[str], list[np.ndarray]]:
    """Forward each protein through ankh-base, return per-residue (L, 768) arrays.

    Shortest-first so a stray OOM on the longest does not lose the batch; per-protein
    forward (batch=1) to bound VRAM; arrays kept in RAM as float32 for the study.
    """
    import torch

    from protea_backends.ankh import AnkhBackend

    be = AnkhBackend()
    noop = lambda *a, **k: None  # noqa: E731
    model, tok = be.load_model(MODEL, "cuda", emit=noop)
    log.info("ankh-base loaded on cuda")

    order = sorted(range(len(proteins)), key=lambda i: proteins[i][1])
    accs: list[str] = []
    res: list[np.ndarray] = []
    t0 = time.time()
    ok = skip = 0
    for n, i in enumerate(order):
        acc, length, seq = proteins[i]
        try:
            t = be._compute_residue_tensors(model, tok, [seq], layers=[0], layer_agg="mean")[0]
            arr = t.detach().float().cpu().numpy().astype(np.float32)
            del t
            accs.append(acc)
            res.append(arr)
            ok += 1
        except torch.cuda.OutOfMemoryError:
            skip += 1
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            skip += 1
            log.warning("skip %s (len %d): %s", acc, length, type(e).__name__)
        if n % 200 == 0:
            torch.cuda.empty_cache()
            log.info("forward %d/%d ok=%d skip=%d (%.0fs)", n, len(order), ok, skip, time.time() - t0)
    torch.cuda.empty_cache()
    log.info("forward DONE: %d residue sets (%d skipped) in %.0fs", ok, skip, time.time() - t0)
    return accs, res


# ---------------------------------------------------------------------------
# Pair sampling (intra- and cross-bucket) + gold computation
# ---------------------------------------------------------------------------
def sample_pairs(buckets: np.ndarray, n_pairs: int, seed: int) -> list[tuple[int, int]]:
    """Sample ``n_pairs`` distinct unordered index pairs (mix intra/cross-bucket)."""
    rng = np.random.default_rng(seed + 1)
    n = len(buckets)
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


def compute_gold(
    res: list[np.ndarray],
    pairs: list[tuple[int, int]],
    *,
    with_ot: bool,
    ot_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Late-interaction max-sim (primary) and Sinkhorn-OT (control) gold per pair.

    Max-sim is run on GPU (each pair is a small Lp x Lq cosine reduction). OT is the
    slower control; to keep it tractable each residue set is strided to <= ``ot_cap``
    positions for the OT arm only (the max-sim gold uses the full sets).
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # pre-normalise + move residue sets to GPU once (float16 to fit; cosine is robust)
    gpu: list[torch.Tensor] = []
    for r in res:
        t = torch.tensor(r, dtype=torch.float32, device=dev)
        t = t / (t.norm(dim=1, keepdim=True) + 1e-12)
        gpu.append(t.half())

    gold = np.empty(len(pairs), dtype=np.float64)
    for n, (i, j) in enumerate(pairs):
        with torch.no_grad():
            sim = (gpu[i].float() @ gpu[j].float().T)
            gold[n] = float(0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean()))
        if n % 4000 == 0:
            torch.cuda.empty_cache()
    del gpu
    torch.cuda.empty_cache()

    ot = np.full(len(pairs), np.nan, dtype=np.float64)
    if with_ot:
        def strided(L: int, cap: int) -> np.ndarray:
            if L <= cap:
                return np.arange(L)
            return np.linspace(0, L - 1, cap).round().astype(int)

        for n, (i, j) in enumerate(pairs):
            pi = res[i][strided(res[i].shape[0], ot_cap)]
            qj = res[j][strided(res[j].shape[0], ot_cap)]
            ot[n] = sinkhorn_ot_sim(pi, qj)
    return gold, ot


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho via rank-Pearson (avoids a scipy dependency in the hot loop)."""
    from scipy.stats import spearmanr

    if len(a) < 30:
        return float("nan")
    rho = spearmanr(a, b)[0]
    return float(rho) if rho == rho else float("nan")


# ---------------------------------------------------------------------------
# The compaction x budget sweep
# ---------------------------------------------------------------------------
def build_compactions(
    res: list[np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    """Materialise every (compaction, budget) variant: codes, sim fn, bytes/protein."""
    d = res[0].shape[1]
    variants: list[dict] = []

    # dense pooled controls (full d floats, stored float16 -> 2*d bytes)
    mean_mat = mean_pool(res)
    variants.append(
        {"compaction": "mean", "budget": f"d{d}", "bytes": 2 * d,
         "sim": cosine_pairwise_fn(mean_mat)}
    )
    max_mat = max_pool(res)
    variants.append(
        {"compaction": "max", "budget": f"d{d}", "bytes": 2 * d,
         "sim": cosine_pairwise_fn(max_mat)}
    )

    # PCA-of-mean at several reduced dims (dense control at smaller budgets)
    for r in args.pca_dims:
        if r >= d:
            continue
        pmat = pca_fit_transform(mean_mat, r)
        variants.append(
            {"compaction": "pca_mean", "budget": f"r{r}", "bytes": 2 * r,
             "sim": cosine_pairwise_fn(pmat)}
        )

    # SDR-union sparse compaction at several final-k (k uint16 indices -> 2k bytes)
    for fk in args.sdr_k:
        sdr, nbytes = sdr_union_codes(res, per_residue_k=args.sdr_per_residue_k, final_k=fk)
        variants.append(
            {"compaction": "sdr_union", "budget": f"k{fk}", "bytes": nbytes,
             "sim": tanimoto_pairwise_fn(sdr)}
        )

    # multi-vector dense control (M residue codes -> 2*M*d bytes)
    for m in args.mv_m:
        codes, nbytes = multivector_codes(res, m, seed=args.seed)
        variants.append(
            {"compaction": "multivector", "budget": f"M{m}", "bytes": nbytes,
             "sim": multivector_pairwise_fn(codes)}
        )

    # ---- selection-vs-matching decomposition -------------------------------
    # Collapse the SAME M representatives the multi-vector arm keeps into a single
    # vector (selection, NO late-interaction matching). If these recover the
    # multi-vector edge over plain `mean`, the edge was the SELECTION (a cheap 2*d
    # flat code suffices); if they sit at `mean`, the edge was the MATCHING.
    for m in args.rep_m:
        reps = representative_residues(res, m, seed=args.seed)
        # mean of the M representatives (selection + flat collapse) -> 2*d bytes
        variants.append(
            {"compaction": "mean_of_M", "budget": f"M{m}", "bytes": 2 * d,
             "sim": cosine_pairwise_fn(mean_of_reps(reps))}
        )
        # norm-weighted mean of the M representatives (cheap salience proxy) -> 2*d
        variants.append(
            {"compaction": "normweighted_mean_of_M", "budget": f"M{m}", "bytes": 2 * d,
             "sim": cosine_pairwise_fn(normweighted_mean_of_reps(reps))}
        )
        # SDR-union over the M representatives (sparse selection) -> 2*final_k bytes
        for fk in args.rep_sdr_k:
            sdr, nbytes = sdr_union_of_reps(
                reps, per_residue_k=args.sdr_per_residue_k, final_k=fk
            )
            variants.append(
                {"compaction": "sdr_union_of_M", "budget": f"M{m}k{fk}", "bytes": nbytes,
                 "sim": tanimoto_pairwise_fn(sdr)}
            )
    return variants


def score_variant(
    variant: dict,
    pairs: list[tuple[int, int]],
    gold: np.ndarray,
    ot: np.ndarray,
    bucket_of: np.ndarray,
) -> dict:
    """Compute compact-sim per pair, then Spearman vs gold/OT per length bucket + all."""
    sim_fn = variant["sim"]
    comp = np.array([sim_fn(i, j) for i, j in pairs], dtype=np.float64)
    pair_bucket = np.array(
        [
            "intra_" + bucket_of[i] if bucket_of[i] == bucket_of[j] else "cross"
            for i, j in pairs
        ]
    )
    row: dict = {
        "compaction": variant["compaction"],
        "budget": variant["budget"],
        "bytes_per_protein": variant["bytes"],
        "_comp": comp,
    }
    groups = {"all": np.ones(len(pairs), dtype=bool)}
    for b in BUCKET_NAMES:
        groups[f"intra_{b}"] = pair_bucket == ("intra_" + b)
    groups["cross"] = pair_bucket == "cross"
    for gname, mask in groups.items():
        if mask.sum() >= 30:
            row[f"spearman_gold_{gname}"] = spearman(comp[mask], gold[mask])
            row[f"n_pairs_{gname}"] = int(mask.sum())
            if not np.isnan(ot).all():
                row[f"spearman_OT_{gname}"] = spearman(comp[mask], ot[mask])
    return row


# ---------------------------------------------------------------------------
# MLflow logging (parent + one nested run per compaction x budget)
# ---------------------------------------------------------------------------
def log_to_mlflow(args, rows, pairs, gold, ot, accs, lens, run_tag):  # noqa: PLR0913
    # MLflow artifacts (Pareto plots + CSV) land in the self-hosted MinIO bucket; the
    # client needs the S3 endpoint + the documented minioadmin creds to PUT them.
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(getattr(args, "experiment", EXPERIMENT))

    with mlflow.start_run(run_name=f"parent-{run_tag}") as parent:
        mlflow.log_params(
            {
                "backbone": MODEL,
                "gold_primary": "late_interaction_maxsim",
                "gold_control": "sinkhorn_ot",
                "n_proteins": len(accs),
                "n_pairs": len(pairs),
                "n_per_bucket": args.n_per_bucket,
                "bucket_edges": "318/969/1959",
                "sdr_per_residue_k": args.sdr_per_residue_k,
                "seed": args.seed,
                "phase": run_tag,
            }
        )
        for b in BUCKET_NAMES:
            mlflow.log_metric(f"n_proteins_{b}", int((lens_bucket(lens) == b).sum()))

        # tidy CSV (compaction x budget x bucket x spearman_gold x spearman_OT x n)
        buf = io.StringIO()
        fieldnames = ["compaction", "budget", "bytes_per_protein", "length_bucket",
                      "spearman_gold", "spearman_OT", "n_pairs"]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            for grp in ["all", *[f"intra_{b}" for b in BUCKET_NAMES], "cross"]:
                if f"spearman_gold_{grp}" not in row:
                    continue
                w.writerow(
                    {
                        "compaction": row["compaction"],
                        "budget": row["budget"],
                        "bytes_per_protein": row["bytes_per_protein"],
                        "length_bucket": grp,
                        "spearman_gold": round(row[f"spearman_gold_{grp}"], 5),
                        "spearman_OT": round(row.get(f"spearman_OT_{grp}", float("nan")), 5),
                        "n_pairs": row.get(f"n_pairs_{grp}", 0),
                    }
                )
        mlflow.log_text(buf.getvalue(), "compaction_quality.csv")

        # nested run per (compaction x budget): metrics keyed by bucket, stepped by bytes
        for row in rows:
            with mlflow.start_run(
                run_name=f"{row['compaction']}-{row['budget']}", nested=True
            ):
                mlflow.set_tags(
                    {
                        "compaction": row["compaction"],
                        "budget": row["budget"],
                        "gold": "lateinteraction",
                    }
                )
                bts = int(row["bytes_per_protein"])
                mlflow.log_metric("bytes_per_protein", bts)
                for grp in ["all", *[f"intra_{b}" for b in BUCKET_NAMES], "cross"]:
                    if f"spearman_gold_{grp}" in row:
                        mlflow.log_metric(f"spearman_gold_{grp}", row[f"spearman_gold_{grp}"], step=bts)
                        mlflow.log_metric(f"n_pairs_{grp}", row[f"n_pairs_{grp}"], step=bts)
                    if f"spearman_OT_{grp}" in row:
                        mlflow.log_metric(f"spearman_OT_{grp}", row[f"spearman_OT_{grp}"], step=bts)

        # Pareto plot: preservation (all-pairs gold spearman) vs bytes, per compaction
        fig, ax = plt.subplots(figsize=(7, 5))
        comps = sorted({r["compaction"] for r in rows})
        for c in comps:
            pts = sorted(
                [(r["bytes_per_protein"], r["spearman_gold_all"]) for r in rows
                 if r["compaction"] == c and "spearman_gold_all" in r]
            )
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, marker="o", label=c)
        ax.set_xscale("log")
        ax.set_xlabel("bytes per protein (log)")
        ax.set_ylabel("Spearman(compact-sim, gold max-sim), all pairs")
        ax.set_title("Compaction quality Pareto: preservation vs storage budget")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _log_fig(mlflow, plt, fig, "pareto_all.png")

        # Per-bucket Pareto (the length-stratified story; long buckets are the test)
        for b in BUCKET_NAMES:
            key = f"spearman_gold_intra_{b}"
            fig, ax = plt.subplots(figsize=(7, 5))
            any_pt = False
            for c in comps:
                pts = sorted(
                    [(r["bytes_per_protein"], r[key]) for r in rows
                     if r["compaction"] == c and key in r]
                )
                if pts:
                    any_pt = True
                    xs, ys = zip(*pts, strict=True)
                    ax.plot(xs, ys, marker="o", label=c)
            if any_pt:
                ax.set_xscale("log")
                ax.set_xlabel("bytes per protein (log)")
                ax.set_ylabel(f"Spearman vs gold, intra-{b} pairs")
                ax.set_title(f"Compaction quality Pareto -- {b} proteins")
                ax.legend()
                ax.grid(True, alpha=0.3)
                _log_fig(mlflow, plt, fig, f"pareto_{b}.png")
            else:
                plt.close(fig)

        # scatter (compact-sim vs gold) for each compaction at its largest budget
        for c in comps:
            cand = [r for r in rows if r["compaction"] == c and "_comp" in r]
            if not cand:
                continue
            r = max(cand, key=lambda x: x["bytes_per_protein"])
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(r["_comp"], gold, s=2, alpha=0.25)
            ax.set_xlabel(f"{c} compact-sim")
            ax.set_ylabel("gold max-sim")
            ax.set_title(f"{c} @ {r['budget']} ({r['bytes_per_protein']}B)")
            _log_fig(mlflow, plt, fig, f"scatter_{c}.png")

        return parent.info.run_id


def _log_fig(mlflow, plt, fig, name):
    path = Path("/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad") / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))


def lens_bucket(lens: np.ndarray) -> np.ndarray:
    return np.array([length_bucket(int(x)) for x in lens])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bucket", type=int, default=1500)
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca-dims", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--sdr-k", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--sdr-per-residue-k", type=int, default=16)
    ap.add_argument("--mv-m", type=int, nargs="+", default=[2, 8, 32])
    # selection-vs-matching decomposition: collapse the SAME M reps to one vector
    ap.add_argument("--rep-m", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--rep-sdr-k", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--ot-cap", type=int, default=64)
    ap.add_argument("--no-ot", action="store_true")
    ap.add_argument("--experiment", default=EXPERIMENT,
                    help="MLflow experiment (use compaction-aggregate for the decomposition)")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    log.info("=== compaction-quality %s: sampling %d/bucket ===", args.tag, args.n_per_bucket)
    proteins = sample_stratified(args.n_per_bucket, args.seed)
    accs, res = forward_residues(proteins)
    lens = np.array([r.shape[0] for r in res])
    bucket_of = lens_bucket(lens)
    for b in BUCKET_NAMES:
        log.info("forwarded bucket %-10s: %d proteins", b, int((bucket_of == b).sum()))

    pairs = sample_pairs(bucket_of, args.n_pairs, args.seed)
    log.info("sampling %d pairs; computing gold (max-sim%s)...", len(pairs),
             "" if args.no_ot else " + Sinkhorn-OT")
    t0 = time.time()
    gold, ot = compute_gold(res, pairs, with_ot=not args.no_ot, ot_cap=args.ot_cap)
    log.info("gold computed in %.0fs (gold range %.3f..%.3f)", time.time() - t0,
             float(gold.min()), float(gold.max()))

    log.info("building compactions + budgets...")
    variants = build_compactions(res, args)
    rows = []
    for v in variants:
        row = score_variant(v, pairs, gold, ot, bucket_of)
        rows.append(row)
        log.info(
            "  %-12s %-6s %7dB | all=%.4f short=%.4f med=%.4f long=%.4f vlong=%.4f cross=%.4f",
            row["compaction"], row["budget"], row["bytes_per_protein"],
            row.get("spearman_gold_all", float("nan")),
            row.get("spearman_gold_intra_short", float("nan")),
            row.get("spearman_gold_intra_medium", float("nan")),
            row.get("spearman_gold_intra_long", float("nan")),
            row.get("spearman_gold_intra_very_long", float("nan")),
            row.get("spearman_gold_cross", float("nan")),
        )

    run_id = log_to_mlflow(args, rows, pairs, gold, ot, accs, lens, args.tag)
    log.info("MLflow parent run: %s/#/experiments (run_id=%s)", MLFLOW_URI, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
