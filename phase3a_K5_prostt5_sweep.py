"""v27-binary-multiseed sweep: 6 NK+LK cells x 3 seeds = 18 runs.

Replicates v26-binary config (binary objective, lean+lin+emb, neg_pos_ratio=10)
across seeds 42, 137, 244 for all NK+LK cells on bench-v1-K5-v226-lineage-prostt5.
Seeds match the v27 target set; compare against v22 (LB.2) baseline for thesis Ch.6.

Writes:
  - runs/phase3a_K5_prostt5/<cell>_seed<n>/run.json (training artefact)
  - runs/phase3a_K5_prostt5/<cell>_seed<n>/cafaeval_metrics.json
  - runs/phase3a_K5_prostt5/cis.json (per-cell bootstrap 95% CIs)
  - runs/phase3a_K5_prostt5/paired_ci.json (v27 vs v22 paired bootstrap)
  - experiments/v27/multiseed_summary.md (publishable summary table)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

# Paths
WORKTREE = Path(__file__).resolve().parent
REPO_DIR = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab")
REPO_DATASETS = REPO_DIR / "datasets"
DATASET_DIR = REPO_DATASETS / "bench-v1-K5-v226-lineage-prostt5"
OBO_PATH = REPO_DATASETS / "bench-v1-K5" / "go.obo"
RUNS_ROOT = WORKTREE / "runs" / "phase3a_K5_prostt5"
EXPERIMENTS_OUT = WORKTREE / "experiments" / "v27"

# Python binaries
LAB_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
PROTEA_PYTHON = Path("/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python")

# Config
CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco"]
SEEDS = [42, 137, 244]

# v26-binary spec: binary objective, lean+lin+emb (no drop_features), neg_pos_ratio=10
# Source: experiments/_generated/study_v26_binary/ (confirmed from spec_catalog.md)
V26_HPARAMS = {
    "objective": "binary",
    "num_boost_round": 10000,
    "early_stopping_rounds": 100,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "drop_features": [],  # lean+lin+emb = all 56 features; no drops
}

ASPECT_TO_NS = {
    "bpo": "biological_process",
    "mfo": "molecular_function",
    "cco": "cellular_component",
}

# v22 LB.2 per-seed cafaeval Fmax (seeds 42, 7, 137)
# Source: lb3_paired_ci.py _LB2_DATA (canonical, 2026-05-17)
# We use seed ordering [42, 7, 137] for v22; v27 uses [42, 137, 244].
# For the paired CI we treat these as independent (different seed sets),
# so we compare v27 mean +- CI vs v22 mean +- CI (not paired by seed).
V22_CAFAEVAL_FMAX = {
    "nk-mfo": [0.7112, 0.7041, 0.7041],
    "nk-bpo": [0.5599, 0.5571, 0.5618],
    "nk-cco": [0.7733, 0.7830, 0.7758],
    "lk-mfo": [0.6877, 0.6786, 0.6757],
    "lk-bpo": [0.6472, 0.6421, 0.6485],
    "lk-cco": [0.7434, 0.7252, 0.7417],
}

# v26-binary single-seed (seed=42) cafaeval Fmax
# Source: runs/study_v26_binary/cafaeval/results.csv
V26_SINGLE_SEED_FMAX = {
    "nk-mfo": 0.7376,
    "nk-bpo": 0.5848,
    "nk-cco": 0.7992,
    "lk-mfo": 0.6915,
    "lk-bpo": 0.6639,
    "lk-cco": 0.7948,
}

# KNN baseline cafaeval Fmax (fixed, from spec_catalog.md)
KNN_BASELINE_FMAX = {
    "nk-bpo": 0.5333,
    "nk-cco": 0.7000,
    "nk-mfo": 0.6447,
    "lk-bpo": 0.5844,
    "lk-cco": 0.7053,
    "lk-mfo": 0.5816,
}


def _run_dir(cell: str, seed: int) -> Path:
    return RUNS_ROOT / f"{cell}_seed{seed}"


def _make_spec_yaml(cell: str, seed: int, out_dir: Path) -> Path:
    spec = {
        "schema_version": "v1",
        "name": f"phase3a_K5_prostt5_{cell}_seed{seed}",
        "dataset": {
            "manifest": str(DATASET_DIR / "manifest.json"),
        },
        "model": {
            "kind": "lgbm_reranker",
            "defaults": {
                **V26_HPARAMS,
            },
        },
        "training": {
            "cell": cell,
            "val_strategy": "protein_group",
            "val_fraction": 0.2,
            "seed": seed,
            "propagate_labels": False,
            "neg_pos_ratio": 10,
        },
        "sweep": {"backend": "none"},
        "output_dir": str(out_dir),
        "tags": [
            "bench-v1-K5-v226-lineage-prostt5",
            cell,
            f"seed{seed}",
            "study_phase3a_K5_prostt5",
            "binary_objective",
            "lean_lin_emb",
            "neg_pos_10",
        ],
        "keep_staging": False,
    }
    spec_path = out_dir / "spec_input.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


def run_training(cell: str, seed: int) -> dict:
    out_dir = _run_dir(cell, seed)
    run_json = out_dir / "run.json"

    if run_json.exists():
        try:
            report = json.loads(run_json.read_text())
            if report.get("status") == "ok":
                fmax = report.get("metrics", {}).get("test_fmax", 0)
                print(f"  [skip] {cell} seed={seed} already done (test_fmax={fmax:.4f})")
                return report
        except Exception:
            pass

    spec_path = _make_spec_yaml(cell, seed, out_dir)
    print(f"  [train] {cell} seed={seed} -> {out_dir}", flush=True)
    t0 = time.monotonic()

    run_script = REPO_DIR / "scripts" / "run.py"
    proc = subprocess.run(
        [str(LAB_PYTHON), str(run_script), str(spec_path),
         "--datasets-root", str(REPO_DATASETS)],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
        timeout=5400,  # 90 min hard limit per run
    )
    dur = time.monotonic() - t0

    if proc.returncode != 0:
        print(f"  [FAIL] {cell} seed={seed} exit={proc.returncode} in {dur:.0f}s")
        print(f"  stderr: {proc.stderr[-800:]}")
        return {"status": "fail", "duration_s": dur}

    try:
        report = json.loads(run_json.read_text())
        fmax = report.get("metrics", {}).get("test_fmax", 0)
        print(f"  [done] {cell} seed={seed} test_fmax={fmax:.4f} dur={dur/60:.1f}min")
        return report
    except Exception as exc:
        print(f"  [FAIL] {cell} seed={seed} could not read run.json: {exc}")
        return {"status": "fail", "duration_s": dur}


# cafaeval
_DRIVER_SRC = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    prop="fill", norm="cafa", no_orphans=True,
    max_terms=500, th_step=0.001, n_cpu=1,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


def _extract_alignment(cell: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive (protein, go, label) in bucket-sort order for a cell."""
    sys.path.insert(0, str(REPO_DIR / "src"))
    from protea_reranker_lab.data import iter_batches

    cat, asp = cell.split("-", 1)
    prots, gos, labs = [], [], []
    for batch in iter_batches(
        DATASET_DIR / "eval.parquet",
        columns=["protein_accession", "go_term_id", "label"],
        category=cat, aspect=asp, batch_size=200_000,
    ):
        prots.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        gos.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))
        labs.append(batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False))
    proteins = np.concatenate(prots)
    go_terms = np.concatenate(gos)
    labels = np.concatenate(labs)

    buckets = np.fromiter(
        (zlib.crc32(p.encode("ascii")) % 32 for p in proteins),
        count=len(proteins), dtype=np.int32,
    )
    order = np.lexsort((proteins, buckets))
    return proteins[order], go_terms[order], labels[order]


def run_cafaeval(cell: str, seed: int, run_report: dict) -> dict:
    out_dir = _run_dir(cell, seed)
    pred_pq = out_dir / "predictions.parquet"
    cafaeval_json = out_dir / "cafaeval_metrics.json"

    if cafaeval_json.exists():
        try:
            m = json.loads(cafaeval_json.read_text())
            if m.get("cafaeval_fmax") is not None:
                print(f"  [skip cafaeval] {cell} seed={seed} fmax={m['cafaeval_fmax']:.4f}")
                return m
        except Exception:
            pass

    if not pred_pq.exists():
        print(f"  [skip cafaeval] {cell} seed={seed}: no predictions.parquet")
        return {}

    print(f"  [cafaeval] {cell} seed={seed}", flush=True)

    pred_t = pq.read_table(str(pred_pq), columns=["protein_accession", "label", "score"])
    proteins_p = pred_t.column("protein_accession").to_numpy(zero_copy_only=False)
    labels_p = pred_t.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
    scores = pred_t.column("score").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    proteins_a, gos, labels_a = _extract_alignment(cell)
    if not np.array_equal(proteins_p, proteins_a):
        print(f"  [err] {cell} seed={seed}: protein alignment mismatch")
        return {}
    if not np.array_equal(labels_p, labels_a):
        print(f"  [err] {cell} seed={seed}: label mismatch")
        return {}

    cell_dir = out_dir / "cafaeval"
    cell_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = cell_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)

    pred_tsv = cell_dir / "pred.tsv"
    gt_tsv = cell_dir / "gt.tsv"
    with pred_tsv.open("w") as f:
        for p, g, s in zip(proteins_p, gos, scores):
            f.write(f"{p}\t{g}\t{s:.6f}\n")
    pos = labels_p > 0
    with gt_tsv.open("w") as f:
        for p, g in zip(proteins_p[pos], gos[pos]):
            f.write(f"{p}\t{g}\n")

    pred_dst = pred_dir / f"{cell}.tsv"
    pred_dst.write_bytes(pred_tsv.read_bytes())

    raw_json = cell_dir / "_raw_metrics.json"
    driver = cell_dir / "_driver.py"
    driver.write_text(_DRIVER_SRC.format(
        obo=str(OBO_PATH),
        pred_dir=str(pred_dir),
        gt=str(gt_tsv),
        out_json=str(raw_json),
    ))

    proc = subprocess.run(
        [str(PROTEA_PYTHON), str(driver)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(f"  [FAIL cafaeval] {cell} seed={seed}: {proc.stderr[-400:]}")
        return {}

    raw = json.loads(raw_json.read_text())
    aspect = cell.split("-", 1)[1]
    target_ns = ASPECT_TO_NS.get(aspect)
    cafaeval_fmax = None

    for metric_kind, records in raw.items():
        for rec in records:
            ns = rec.get("ns") or rec.get("namespace") or ""
            if ns == target_ns:
                for k in ("f", "Fmax", "fmax"):
                    if k in rec and rec[k] is not None:
                        try:
                            cafaeval_fmax = float(rec[k])
                        except (TypeError, ValueError):
                            pass
                        break
            if cafaeval_fmax is not None:
                break
        if cafaeval_fmax is not None:
            break

    if cafaeval_fmax is None:
        for metric_kind, records in raw.items():
            for rec in records:
                for k in ("f", "Fmax", "fmax"):
                    if k in rec and rec[k] is not None:
                        try:
                            cafaeval_fmax = float(rec[k])
                        except (TypeError, ValueError):
                            pass
                        if cafaeval_fmax is not None:
                            break
                if cafaeval_fmax is not None:
                    break
            if cafaeval_fmax is not None:
                break

    n_pos = int(pos.sum())
    metrics = {
        "cell": cell,
        "seed": seed,
        "cafaeval_fmax": cafaeval_fmax,
        "n_rows": len(proteins_p),
        "n_positives": n_pos,
    }
    cafaeval_json.write_text(json.dumps(metrics, indent=2))
    print(f"  [cafaeval done] {cell} seed={seed} cafaeval_fmax={cafaeval_fmax}")
    return metrics


def bootstrap_ci(
    vals: np.ndarray,
    n_iter: int = 10000,
    seed: int = 0,
) -> dict:
    """Bootstrap 95% CI on the mean of `vals`."""
    rng = np.random.default_rng(seed)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_iter)
    ])
    ci_lo = float(np.quantile(boot_means, 0.025))
    ci_hi = float(np.quantile(boot_means, 0.975))
    return {
        "n_seeds": len(vals),
        "values": vals.tolist(),
        "mean": mean,
        "std": std,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_half_width": (ci_hi - ci_lo) / 2,
    }


def compute_v22_vs_v27_comparison(
    v27_cell_fmax: dict[str, list[float]],
    n_iter: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute per-cell comparison: v27-binary vs v22-lambdarank (LB.2).

    Since the two runs use different seed sets (v22: 42/7/137; v27: 42/137/244),
    the arms are not paired by seed. We compute independent bootstrap CIs on
    each arm and report delta = v27_mean - v22_mean + [v27_CI_lo - v22_CI_hi,
    v27_CI_hi - v22_CI_lo] as a conservative interval.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for cell in CELLS:
        v27_vals = np.array(
            [v for v in v27_cell_fmax.get(cell, []) if v is not None],
            dtype=float,
        )
        v22_vals = np.array(V22_CAFAEVAL_FMAX[cell], dtype=float)

        if len(v27_vals) == 0:
            results[cell] = {"v27_mean": None, "v22_mean": float(v22_vals.mean())}
            continue

        # Independent bootstrap for each arm
        boot_v27 = np.array([
            rng.choice(v27_vals, size=len(v27_vals), replace=True).mean()
            for _ in range(n_iter)
        ])
        boot_v22 = np.array([
            rng.choice(v22_vals, size=len(v22_vals), replace=True).mean()
            for _ in range(n_iter)
        ])
        # Delta distribution: v27 - v22 (independent arms, same bootstrap draws)
        boot_delta = boot_v27 - boot_v22

        results[cell] = {
            "v27_mean": float(v27_vals.mean()),
            "v27_ci_lo": float(np.quantile(boot_v27, 0.025)),
            "v27_ci_hi": float(np.quantile(boot_v27, 0.975)),
            "v27_ci_half_width": float((np.quantile(boot_v27, 0.975) - np.quantile(boot_v27, 0.025)) / 2),
            "v22_mean": float(v22_vals.mean()),
            "v22_ci_lo": float(np.quantile(boot_v22, 0.025)),
            "v22_ci_hi": float(np.quantile(boot_v22, 0.975)),
            "delta_mean": float(boot_delta.mean()),
            "delta_ci_lo": float(np.quantile(boot_delta, 0.025)),
            "delta_ci_hi": float(np.quantile(boot_delta, 0.975)),
            "sig_95": int(np.quantile(boot_delta, 0.025) > 0),
        }
    return results


def write_summary_md(
    cis: dict,
    comparison: dict,
    out_path: Path,
    seed_wall_clock_min: dict[tuple[str, int], float],
) -> None:
    """Write publishable multiseed_summary.md (no em-dashes)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# v27-binary-multiseed: publishable CI summary",
        "",
        "Study: study-v27-binary-multiseed",
        "Hparams: binary objective, lean+lin+emb (56 features), neg_pos_ratio=10,",
        "  num_boost_round=10000, lr=0.05, num_leaves=63, min_data_in_leaf=100,",
        "  early_stopping_rounds=100.",
        "Seeds: 42, 137, 244 (3-seed replication of v26-binary champion).",
        "Eval set: bench-v1-K5-v226-lineage-prostt5 (eval window v226-v230).",
        "Cafaeval: prop=fill, norm=cafa, no_orphans=True.",
        "",
        "## Per-cell cafaeval Fmax: v27-binary (mean +- 95% CI half-width)",
        "",
        "| cell | seed=42 | seed=137 | seed=244 | mean | CI half-width |",
        "|-|-|-|-|-|-|",
    ]
    for cell in CELLS:
        ci = cis.get(cell, {})
        vals = ci.get("values", [])
        mean = ci.get("mean")
        hw = ci.get("ci_half_width")
        s42 = f"{vals[0]:.4f}" if len(vals) > 0 else "N/A"
        s137 = f"{vals[1]:.4f}" if len(vals) > 1 else "N/A"
        s244 = f"{vals[2]:.4f}" if len(vals) > 2 else "N/A"
        mean_s = f"{mean:.4f}" if mean is not None else "N/A"
        hw_s = f"{hw:.4f}" if hw is not None else "N/A"
        lines.append(f"| {cell} | {s42} | {s137} | {s244} | {mean_s} | {hw_s} |")

    lines += [
        "",
        "## v27-binary vs v22-lambdarank (LB.2) comparison",
        "",
        "v22 seeds: 42, 7, 137 (LB.2, leakage-fixed lambdarank, lean+lin features).",
        "v27 seeds: 42, 137, 244 (this run, binary objective, lean+lin+emb features).",
        "Bootstrap: N=10000, independent arms (different seed sets).",
        "sig_95: 1 if delta 95% CI lower bound > 0.",
        "",
        "| cell | v27 mean | v22 mean | delta | delta 95% CI | sig_95 |",
        "|-|-|-|-|-|-|",
    ]
    for cell in CELLS:
        comp = comparison.get(cell, {})
        v27m = comp.get("v27_mean")
        v22m = comp.get("v22_mean")
        dm = comp.get("delta_mean")
        d_lo = comp.get("delta_ci_lo")
        d_hi = comp.get("delta_ci_hi")
        sig = comp.get("sig_95", "?")
        v27_s = f"{v27m:.4f}" if v27m is not None else "N/A"
        v22_s = f"{v22m:.4f}" if v22m is not None else "N/A"
        dm_s = f"{dm:+.4f}" if dm is not None else "N/A"
        ci_s = f"[{d_lo:+.4f}, {d_hi:+.4f}]" if (d_lo is not None and d_hi is not None) else "N/A"
        lines.append(f"| {cell} | {v27_s} | {v22_s} | {dm_s} | {ci_s} | {sig} |")

    lines += [
        "",
        "## v26-binary single-seed vs v27-binary multiseed comparison",
        "",
        "| cell | v26 (seed=42) | v27 mean | delta |",
        "|-|-|-|-|",
    ]
    for cell in CELLS:
        v26 = V26_SINGLE_SEED_FMAX.get(cell)
        ci = cis.get(cell, {})
        v27m = ci.get("mean")
        dm = (v27m - v26) if (v27m is not None and v26 is not None) else None
        v26_s = f"{v26:.4f}" if v26 is not None else "N/A"
        v27_s = f"{v27m:.4f}" if v27m is not None else "N/A"
        dm_s = f"{dm:+.4f}" if dm is not None else "N/A"
        lines.append(f"| {cell} | {v26_s} | {v27_s} | {dm_s} |")

    lines += [
        "",
        "## Training wall-clock (per seed)",
        "",
        "| cell | seed=42 (min) | seed=137 (min) | seed=244 (min) |",
        "|-|-|-|-|",
    ]
    for cell in CELLS:
        t42 = seed_wall_clock_min.get((cell, 42))
        t137 = seed_wall_clock_min.get((cell, 137))
        t244 = seed_wall_clock_min.get((cell, 244))
        def fmt(t):
            return f"{t:.1f}" if t is not None else "N/A"
        lines.append(f"| {cell} | {fmt(t42)} | {fmt(t137)} | {fmt(t244)} |")

    lines += [
        "",
        "## Outcome",
        "",
        "See experiments/v27/multiseed_summary.md for the full CI table.",
        "Artifact root: runs/phase3a_K5_prostt5/",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written -> {out_path}")


def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_OUT.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    print(f"\n=== v27-binary-multiseed: {len(CELLS)} cells x {len(SEEDS)} seeds ===")
    print("Objective: binary | Features: lean+lin+emb (all 56) | neg_pos_ratio=10")
    print(f"Seeds: {SEEDS}\n")

    wall_clock: dict[tuple[str, int], float] = {}

    # Phase 1: Training
    all_reports: dict[str, dict[int, dict]] = {}
    for cell in CELLS:
        all_reports[cell] = {}
        for seed in SEEDS:
            elapsed = time.time() - start_ts
            if elapsed > 3 * 3600:
                print(f"[HALT] 3h wall-clock exceeded at {elapsed/3600:.1f}h. Stopping.")
                sys.exit(1)
            t0 = time.monotonic()
            report = run_training(cell, seed)
            dur_min = (time.monotonic() - t0) / 60
            wall_clock[(cell, seed)] = dur_min
            all_reports[cell][seed] = report

    print("\n=== Phase 1 done: training complete ===\n")

    # Phase 2: cafaeval
    cafaeval_results: dict[str, dict[int, dict]] = {}
    for cell in CELLS:
        cafaeval_results[cell] = {}
        for seed in SEEDS:
            elapsed = time.time() - start_ts
            if elapsed > 3.5 * 3600:
                print("[HALT] 3.5h exceeded. Stopping before cafaeval.")
                sys.exit(1)
            m = run_cafaeval(cell, seed, all_reports[cell].get(seed, {}))
            cafaeval_results[cell][seed] = m

    print("\n=== Phase 2 done: cafaeval complete ===\n")

    # Phase 3: Bootstrap CIs
    cell_seed_fmax: dict[str, dict[int, float | None]] = {}
    for cell in CELLS:
        cell_seed_fmax[cell] = {}
        for seed in SEEDS:
            m = cafaeval_results[cell].get(seed, {})
            cell_seed_fmax[cell][seed] = m.get("cafaeval_fmax")

    cis: dict[str, dict] = {}
    for cell in CELLS:
        vals = np.array(
            [v for v in cell_seed_fmax[cell].values() if v is not None],
            dtype=float,
        )
        if len(vals) == 0:
            cis[cell] = {"mean": None, "ci_half_width": None, "values": []}
            continue
        cis[cell] = bootstrap_ci(vals, n_iter=10000, seed=99)

    cis_path = RUNS_ROOT / "cis.json"
    cis_path.write_text(json.dumps(cis, indent=2))
    print(f"Bootstrap CIs -> {cis_path}")

    # Phase 4: v27 vs v22 comparison
    v27_cell_fmax = {
        cell: [v for v in cell_seed_fmax[cell].values() if v is not None]
        for cell in CELLS
    }
    comparison = compute_v22_vs_v27_comparison(v27_cell_fmax, n_iter=10000, seed=42)
    paired_path = RUNS_ROOT / "paired_ci.json"
    paired_path.write_text(json.dumps(comparison, indent=2))
    print(f"Paired CI -> {paired_path}")

    # Phase 5: Summary
    write_summary_md(cis, comparison, EXPERIMENTS_OUT / "multiseed_summary.md", wall_clock)

    # Full summary JSON
    summary = {
        "task": "phase3a_K5_prostt5",
        "seeds": SEEDS,
        "cells": CELLS,
        "hparams_source": "spec_catalog.md study-v26-binary + experiments/_generated/study_v26_binary/",
        "per_cell_cafaeval_fmax_by_seed": {
            cell: {str(s): cell_seed_fmax[cell].get(s) for s in SEEDS}
            for cell in CELLS
        },
        "per_cell_ci": cis,
        "v27_vs_v22_comparison": comparison,
        "elapsed_s": round(time.time() - start_ts, 1),
    }
    summary_path = RUNS_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON -> {summary_path}")

    # Print final report
    print("\n" + "=" * 70)
    print("v27-binary-multiseed RESULTS")
    print("=" * 70)
    print("Hparams source: spec_catalog.md study-v26-binary")
    print(f"Seeds: {SEEDS}")
    print()
    print("Per-cell cafaeval Fmax (mean +- 95% CI half-width) | vs v22 delta:")
    for cell in CELLS:
        ci = cis[cell]
        comp = comparison.get(cell, {})
        m = ci.get("mean")
        hw = ci.get("ci_half_width")
        dm = comp.get("delta_mean")
        d_lo = comp.get("delta_ci_lo")
        d_hi = comp.get("delta_ci_hi")
        sig = comp.get("sig_95", "?")
        if m is not None and hw is not None:
            m_s = f"{m:.4f} +- {hw:.4f}"
        else:
            m_s = "N/A"
        if dm is not None and d_lo is not None and d_hi is not None:
            vs_s = f"delta={dm:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}] sig95={sig}"
        else:
            vs_s = "N/A"
        print(f"  {cell}: {m_s} | {vs_s}")

    print("=" * 70)


if __name__ == "__main__":
    main()
