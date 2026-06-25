#!/usr/bin/env python
"""Backbone size-vs-signal proxy sweep (Stage A: cached mean substrates ONLY).

The decision this informs: we want to ITERATE the architecture (learned post-PLM head,
chunk/aspect/giants) FAST. The backbone PLM is a substrate. Does a SMALL ESM2 carry
enough functional signal to be a faithful dev substrate, or does model size matter
enough to justify the compute?

Stage A is CHEAP: it never extracts per-residue features and never generates an
embedding. It trains the small champion-recipe learned-mean head (Linear -> top-k real,
hard-neg Lin-contrastive, the d8979601 recipe via ``encoder_ablation.fit_encoder``) on
the ALREADY-CACHED mean vectors of each backbone, then reports the GO-correlation proxy:
Spearman(arm_similarity, GO_semantic_similarity) for Resnik AND Lin, per length bucket.

This is a PROXY (GO-correlation), NOT f_micro_w. It measures how well a backbone's
learned-mean code agrees with GO semantics; it does not measure the end task metric.

Per backbone, two arms are reported per length bucket:

  * ``dense-mean-cosine`` : raw mean embedding -> cosine                       (the substrate's native signal)
  * ``learned``           : champion hard-neg k-WTA head -> Tanimoto           (the signal we will iterate on)

The protein sample is shared across all backbones (one length-balanced draw), so a
backbone's per-bucket Spearman is directly comparable to another's. Length buckets are
restricted to L<=1959 (truncation-clean: the smallest cached ESM2 configs cap at 1022
tokens and every config truncates somewhere, so we keep the comparison on lengths every
backbone can represent without differential truncation skew):

    short  : <= 318
    medium : 319 .. 969
    long   : 970 .. 1959

Read-only on the DB. GPU for the learned head (tiny: minutes each on the cached means).
Logs to MLflow experiment ``backbone-size-sweep``.

Run inside the lab venv with MLflow live::

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python scripts/run_backbone_size_sweep.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from protea_reranker_lab.encoder_ablation import ArmSpec, EncoderAblationSpec, apply_encoder, fit_encoder
from protea_reranker_lab.sdr import (
    GoDag,
    cosine_dense,
    information_content,
    lin_pairwise,
    propagate,
    resnik_pairwise,
    tanimoto_dense,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [backbone-sweep] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backbone-sweep")

EXPERIMENT = "backbone-size-sweep"
ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0 (matches the SDR harnesses)

# Truncation-clean length ceiling: keep the comparison on lengths every cached config
# can represent without differential truncation skew. Buckets are within [0, MAX_LEN].
MAX_LEN = 1959
BUCKETS = [
    ("short", 0, 318),
    ("medium", 319, 969),
    ("long", 970, 1959),
]


@dataclass
class Backbone:
    """One cached mean backbone to sweep."""

    key: str  # stable short label
    config_id: str
    family: str
    param_count: int | None
    note: str = ""


# Cached mean backbones (pooling='mean'), discovered from embedding_config. The small
# ESM2 end (8M / 150M) is present but with NULL display_name/param_count metadata; the
# param_count below is the published model size (filled in here for the size-vs-signal
# curve). The d8979601 learned code is itself a derived config and is NOT a backbone.
BACKBONES: list[Backbone] = [
    Backbone("esm2-8m", "ce061d7f-2526-4da8-965a-fbdfdd5a30ca", "ESM2", 7_500_000,
             "esm2_t6_8M_UR50D, 320d (cache has NULL metadata)"),
    Backbone("esm2-150m", "500a0c59-be09-424d-9d51-b7997629c95a", "ESM2", 148_000_000,
             "esm2_t30_150M_UR50D, 640d (cache has NULL metadata)"),
    Backbone("esmc-300m", "c85d1afe-3f49-4ead-82d9-faaa6efe7a2c", "ESMC", 300_000_000, ""),
    Backbone("ankh-base", "08234f06-ba76-4d7d-aaec-ae601096b4fa", "Ankh", 450_000_000, ""),
    Backbone("esmc-600m", "2bf1e753-022f-44b8-a131-9a90acb4024e", "ESMC", 600_000_000, ""),
    Backbone("esm2-650m", "c2e9dda3-e505-4170-b50d-435a451761ac", "ESM2", 650_000_000, ""),
    Backbone("ankh-large", "238f79b1-3068-4c6f-9013-5cc52b4f662b", "Ankh", 1_900_000_000, ""),
    Backbone("prott5-xl", "db4db5ed-e34a-47af-a9ab-cc0f230b0a8c", "ProtT5", 3_000_000_000, ""),
    Backbone("esm2-3b", "55e43f1c-1a3b-4b1d-88c0-26b433f5f673", "ESM2", 3_000_000_000, ""),
    Backbone("prostt5-xl", "c0ae5b69-d6dc-41cf-a711-1739d3d2e170", "ProstT5", 3_000_000_000, ""),
]


def bucket_of(length: int) -> str | None:
    for name, lo, hi in BUCKETS:
        if lo <= length <= hi:
            return name
    return None


def _parse_vec(text: str) -> np.ndarray:
    return np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared length-balanced sample (accessions + lengths + GO closures), pulled ONCE
# ---------------------------------------------------------------------------
def load_sample(dsn: str, seed: int, per_bucket_cap: int):
    """Length-balanced accession sample within [0, MAX_LEN] + leaf annotations (READ-ONLY)."""
    import psycopg2

    rng = np.random.default_rng(seed)
    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True)
    cur = conn.cursor()

    log.info("scanning lengths for v227-annotated proteins (L<=%d)", MAX_LEN)
    cur.execute(
        """
        SELECT a.acc, length(s.sequence) AS seqlen
        FROM (SELECT DISTINCT protein_accession AS acc
              FROM protein_go_annotation WHERE annotation_set_id = %(ann)s) a
        JOIN protein p ON p.accession = a.acc
        JOIN sequence s ON s.id = p.sequence_id
        WHERE p.sequence_id IS NOT NULL AND length(s.sequence) <= %(maxlen)s
        """,
        {"ann": ANN_SET, "maxlen": MAX_LEN},
    )
    cand = cur.fetchall()
    log.info("  %d annotated proteins with a sequence in range", len(cand))

    acc_len = {acc: int(ln) for acc, ln in cand}
    by_bucket: dict[str, list[str]] = {name: [] for name, _, _ in BUCKETS}
    for acc, ln in cand:
        b = bucket_of(int(ln))
        if b is not None:
            by_bucket[b].append(acc)
    sample: list[str] = []
    for name in by_bucket:
        accs_b = by_bucket[name]
        rng.shuffle(accs_b)
        take = accs_b[:per_bucket_cap]
        log.info("  bucket %-7s available=%d -> taking %d", name, len(accs_b), len(take))
        sample.extend(take)
    rng.shuffle(sample)
    sample_t = tuple(sample)

    log.info("pulling v227 leaf annotations for the sample")
    cur.execute(
        """
        SELECT pga.protein_accession, gt.go_id
        FROM protein_go_annotation pga
        JOIN go_term gt ON gt.id = pga.go_term_id
        WHERE pga.annotation_set_id = %s AND pga.protein_accession IN %s
        """,
        (ANN_SET, sample_t),
    )
    leaves: dict[str, list[str]] = {}
    for a, go in cur.fetchall():
        leaves.setdefault(a, []).append(go)

    cur.close()
    conn.close()
    accs = [a for a in sample if a in leaves]
    log.info("  %d sampled accessions with annotations", len(accs))
    return accs, acc_len, leaves


def pull_mean(dsn: str, accs: list[str], config_id: str) -> dict[str, np.ndarray]:
    """accession -> mean-pooled embedding (over chunks) for a config (READ-ONLY).

    The mean configs in scope are single-row per protein, but ESM2-3B uses chunking, so
    we average over the rows ordered by chunk_index to be safe across configs.
    """
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.accession, se.chunk_index_s, se.embedding::text
        FROM protein p
        JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
        WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)
        ORDER BY p.accession, se.chunk_index_s
        """,
        (config_id, list(accs)),
    )
    tmp: dict[str, list[np.ndarray]] = {}
    for acc, _ci, vtext in cur:
        tmp.setdefault(acc, []).append(_parse_vec(vtext))
    cur.close()
    conn.close()
    return {a: np.vstack(v).mean(0).astype(np.float32) for a, v in tmp.items()}


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


# ---------------------------------------------------------------------------
# One backbone: train the learned head on its means, score dense + learned per bucket
# ---------------------------------------------------------------------------
def run_backbone(bb: Backbone, accs, closures_all, lengths, buckets, dag, args) -> dict:
    log.info("=== backbone %s (%s, %s params) ===", bb.key, bb.family,
             f"{bb.param_count:,}" if bb.param_count else "NA")
    mean = pull_mean(args.dsn, accs, bb.config_id)
    have = [i for i, a in enumerate(accs) if a in mean]
    if len(have) < args.min_bucket_n:
        log.warning("  backbone %s: only %d means pulled -> skipped", bb.key, len(have))
        return {"param_count": bb.param_count, "family": bb.family, "dim": None,
                "n": len(have), "skipped": True}
    sub_accs = [accs[i] for i in have]
    sub_clo = [closures_all[i] for i in have]
    X = np.vstack([mean[a] for a in sub_accs]).astype(np.float32)
    dim = int(X.shape[1])
    log.info("  pulled %d means (dim=%d); training learned head", len(sub_accs), dim)

    # Train the champion-recipe hard-neg head on THIS backbone's means (the d8979601
    # recipe). The encoder is small; the means are cached, so this is minutes on GPU.
    arm = ArmSpec(name="learned-k128-hardneg", kind="learned", dict_dim=args.dict_dim,
                  top_k=args.top_k, objective="hard-neg")
    spec = EncoderAblationSpec(
        name=f"backbone-sweep-{bb.key}", embedding_config_id=bb.config_id,
        annotation_set_id=ANN_SET, epochs=args.epochs, train_pairs=args.train_pairs,
        knn=args.knn, seed=args.seed,
    )
    enc = fit_encoder(X, sub_clo, dag, arm, spec)
    codes = apply_encoder(enc, X, args.top_k)  # (n, dict_dim) top-k real
    code_bits = (codes != 0.0).astype(np.uint8)
    k_active = int(np.median(code_bits.sum(axis=1))) or 1

    rng = np.random.default_rng(args.seed)
    table: dict[str, dict] = {}
    for bname, all_idx in buckets.items():
        idxs = [have.index(i) for i in all_idx if i in have]
        n = len(idxs)
        entry: dict[str, float | int | None] = {"N": n}
        if n < args.min_bucket_n:
            log.warning("  bucket %-7s N=%d below min -> skipped", bname, n)
            table[bname] = entry
            continue
        bclo = [sub_clo[i] for i in idxs]
        ic = information_content(bclo, dag)
        best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in bclo]
        pairs = sample_pairs(n, args.n_pairs, rng)
        ii = np.array([p[0] for p in pairs])
        jj = np.array([p[1] for p in pairs])
        go = {
            "resnik": np.asarray(resnik_pairwise(bclo, ic, pairs), dtype=np.float64),
            "lin": np.asarray(lin_pairwise(bclo, ic, pairs, best_ic), dtype=np.float64),
        }
        # dense-mean-cosine
        dense = X[idxs]
        cos = cosine_dense(dense)[ii, jj]
        # learned Tanimoto
        lbits = code_bits[idxs]
        ltan = tanimoto_dense(lbits, k_active)[ii, jj]
        for m in ("resnik", "lin"):
            entry[f"dense-mean-cosine|{m}"] = float(spearmanr(cos, go[m])[0])
            entry[f"learned|{m}"] = float(spearmanr(ltan, go[m])[0])
        log.info("  bucket %-7s N=%d pairs=%d dense(resnik)=%.4f learned(resnik)=%.4f",
                 bname, n, len(pairs), entry["dense-mean-cosine|resnik"],
                 entry["learned|resnik"])
        table[bname] = entry

    return {"param_count": bb.param_count, "family": bb.family, "dim": dim,
            "n": len(sub_accs), "k_active": k_active, "table": table, "skipped": False}


def run(args: argparse.Namespace) -> dict:
    dag = GoDag.from_obo(Path(args.obo).expanduser())
    accs, acc_len, leaves = load_sample(args.dsn, args.seed, args.per_bucket_cap)
    closures_all = [propagate(leaves[a], dag) for a in accs]
    keep = [i for i, c in enumerate(closures_all) if c]
    accs = [accs[i] for i in keep]
    closures_all = [closures_all[i] for i in keep]
    lengths = np.array([acc_len[a] for a in accs])
    log.info("shared sample: %d proteins with a non-empty closure", len(accs))

    buckets: dict[str, list[int]] = {name: [] for name, _, _ in BUCKETS}
    for i, ln in enumerate(lengths):
        b = bucket_of(int(ln))
        if b is not None:
            buckets[b].append(i)
    for name, _, _ in BUCKETS:
        log.info("  bucket %-7s N=%d", name, len(buckets[name]))

    only = set(args.only) if args.only else None
    results: dict[str, dict] = {}
    for bb in BACKBONES:
        if only and bb.key not in only:
            continue
        results[bb.key] = run_backbone(bb, accs, closures_all, lengths, buckets, dag, args)

    return {
        "n_proteins": len(accs),
        "buckets": [b[0] for b in BUCKETS],
        "max_len": MAX_LEN,
        "results": results,
        "seed": args.seed,
        "dict_dim": args.dict_dim,
        "top_k": args.top_k,
        "epochs": args.epochs,
    }


# ---------------------------------------------------------------------------
# Artifacts + report
# ---------------------------------------------------------------------------
def _flatten_rows(result: dict) -> list[dict]:
    rows = []
    for key, r in result["results"].items():
        if r.get("skipped"):
            continue
        for bname in result["buckets"]:
            entry = r.get("table", {}).get(bname, {})
            row = {
                "backbone": key, "family": r.get("family"),
                "param_count": r.get("param_count"), "dim": r.get("dim"),
                "bucket": bname, "N": entry.get("N", 0),
            }
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
    csv = out_dir / "backbone_size_sweep.csv"
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(
            (f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            for c in cols
        ))
    csv.write_text("\n".join(lines) + "\n")
    paths.append(csv)
    js = out_dir / "backbone_size_sweep.json"
    js.write_text(json.dumps(result, indent=2))
    paths.append(js)
    return paths


def _metric_name(bucket: str, arm: str, metric: str) -> str:
    """Human-readable MLflow metric key, e.g. ``spearman_learned_resnik_long``.

    ``arm`` in {dense-mean-cosine, learned}; ``metric`` in {resnik, lin}.
    """
    arm_h = {"dense-mean-cosine": "dense", "learned": "learned"}.get(arm, arm)
    return f"spearman_{arm_h}_{metric}_{bucket}"


def log_to_mlflow(args, result, artifacts) -> str | None:
    """Log a parent sweep run + one NESTED child run per backbone, readable in the UI.

    Layout in the ``backbone-size-sweep`` experiment:

      * parent run ``sweep-all-backbones`` holds the shared setup (window, OBO, sample,
        head recipe) as params, the full CSV/JSON artifacts, and a SIZE-vs-SIGNAL series
        (``size_signal_curve_resnik_long`` etc.) stepped by param_count so the MLflow
        chart draws Spearman against model size directly.
      * each backbone is a child run named by what it is (e.g. ``esm2-8m``) with its
        family / param_count / dim as both params AND tags (filterable), and one readable
        metric per (bucket x arm x GO-metric), e.g. ``spearman_learned_resnik_long``.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return None
    try:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="sweep-all-backbones") as parent:
            mlflow.set_tags({
                "stage": "A-cached-mean-substrates",
                "proxy": "GO-correlation (Resnik+Lin), NOT f_micro_w",
                "question": "size-vs-signal: cheapest faithful dev substrate",
                "head_recipe": "champion hard-neg k-WTA (d8979601)",
            })
            mlflow.log_params({
                "annotation_set": ANN_SET, "window": "v227-t0", "max_len": MAX_LEN,
                "n_proteins": result["n_proteins"], "n_pairs_per_bucket": args.n_pairs,
                "head_dict_dim": args.dict_dim, "head_top_k": args.top_k,
                "head_epochs": args.epochs, "head_train_pairs": args.train_pairs,
                "head_knn_hardneg": args.knn, "seed": args.seed,
                "buckets": ",".join(result["buckets"]),
                "backbones": ",".join(result["results"].keys()),
            })
            for p in artifacts:
                mlflow.log_artifact(str(p))

            # Size-vs-signal curve: one point per backbone, stepped by param_count so the
            # MLflow metric chart shows Spearman climbing/plateauing with model size.
            ordered = sorted(
                ((k, r) for k, r in result["results"].items()
                 if not r.get("skipped") and r.get("param_count")),
                key=lambda kr: kr[1]["param_count"],
            )
            for key, r in ordered:
                step = int(r["param_count"])
                for bname in result["buckets"]:
                    entry = r.get("table", {}).get(bname, {})
                    for arm in ("dense-mean-cosine", "learned"):
                        for metric in ("resnik", "lin"):
                            v = entry.get(f"{arm}|{metric}")
                            if isinstance(v, float):
                                arm_h = "dense" if arm == "dense-mean-cosine" else "learned"
                                mlflow.log_metric(
                                    f"size_signal_curve_{arm_h}_{metric}_{bname}", v, step=step)

            parent_id = parent.info.run_id

            # One nested child run per backbone (a browsable, filterable row in the UI).
            for key, r in result["results"].items():
                if r.get("skipped"):
                    continue
                with mlflow.start_run(run_name=key, nested=True):
                    mlflow.set_tags({
                        "backbone": key, "family": r.get("family") or "NA",
                        "param_count": str(r.get("param_count") or "NA"),
                        "dim": str(r.get("dim") or "NA"),
                        "stage": "A-cached-mean-substrates",
                    })
                    mlflow.log_params({
                        "backbone": key, "family": r.get("family"),
                        "param_count": r.get("param_count"), "dim": r.get("dim"),
                        "k_active_learned": r.get("k_active"),
                        "n_proteins": r.get("n"),
                    })
                    for bname, entry in r.get("table", {}).items():
                        if isinstance(entry.get("N"), int):
                            mlflow.log_metric(f"n_proteins_{bname}", entry["N"])
                        for arm in ("dense-mean-cosine", "learned"):
                            for metric in ("resnik", "lin"):
                                v = entry.get(f"{arm}|{metric}")
                                if isinstance(v, float):
                                    mlflow.log_metric(_metric_name(bname, arm, metric), v)
                        # learned uplift over the dense substrate (the iterate-on signal)
                        for metric in ("resnik", "lin"):
                            d = entry.get(f"dense-mean-cosine|{metric}")
                            lrn = entry.get(f"learned|{metric}")
                            if isinstance(d, float) and isinstance(lrn, float):
                                mlflow.log_metric(f"learned_uplift_{metric}_{bname}", lrn - d)
        log.info("mlflow: parent run %s (+%d backbone children) -> %r",
                 parent_id, sum(1 for r in result["results"].values() if not r.get("skipped")),
                 EXPERIMENT)
        return parent_id
    except Exception as exc:  # pragma: no cover
        log.warning("mlflow logging failed (%s)", exc)
        return None


def _print_table(result: dict) -> None:
    print("\n=== backbone x bucket Spearman(arm, GO) | learned head (hard-neg k-WTA) + dense baseline ===")
    print("PROXY = GO-semantic correlation (Resnik + Lin), NOT f_micro_w.\n")
    header = (f"{'backbone':<12}{'params':>14}{'dim':>6}{'bucket':>8}{'N':>7}"
              f"{'dense|res':>11}{'learn|res':>11}{'dense|lin':>11}{'learn|lin':>11}")
    print(header)
    print("-" * len(header))
    for key, r in result["results"].items():
        if r.get("skipped"):
            print(f"{key:<12}{'(skipped)':>14}")
            continue
        pc = f"{r['param_count']:,}" if r.get("param_count") else "NA"
        for bname in result["buckets"]:
            entry = r.get("table", {}).get(bname, {})
            def g(k):
                v = entry.get(k)
                return f"{v:>11.4f}" if isinstance(v, float) else f"{'-':>11}"
            print(f"{key:<12}{pc:>14}{str(r.get('dim')):>6}{bname:>8}{entry.get('N', 0):>7}"
                  f"{g('dense-mean-cosine|resnik')}{g('learned|resnik')}"
                  f"{g('dense-mean-cosine|lin')}{g('learned|lin')}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    p.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    p.add_argument("--per-bucket-cap", type=int, default=1500,
                   help="max proteins per length bucket in the shared sample")
    p.add_argument("--n-pairs", type=int, default=20_000, help="protein pairs per bucket")
    p.add_argument("--dict-dim", type=int, default=2048, help="learned code width")
    p.add_argument("--top-k", type=int, default=128, help="k-WTA active count of the learned code")
    p.add_argument("--epochs", type=int, default=120, help="learned head training epochs")
    p.add_argument("--train-pairs", type=int, default=200_000, help="contrastive pairs for the head")
    p.add_argument("--knn", type=int, default=30, help="hard-neg mining neighbourhood")
    p.add_argument("--min-bucket-n", type=int, default=50,
                   help="buckets below this protein count are skipped + flagged")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="+", default=None, help="restrict to these backbone keys")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "backbone_sweep_out"
    artifacts = write_artifacts(result, out_dir)
    run_id = None if args.no_mlflow else log_to_mlflow(args, result, artifacts)
    _print_table(result)
    if run_id:
        print(f"\nMLflow run: {run_id}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
