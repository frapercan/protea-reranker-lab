"""Aggregate the multi-seed ProtST A/B into a per-cell across-seed table + verdict.

Reads the 4 seeds (42 = existing armA_9cell.json/armB_9cell.json; 7/123/2024 =
per-seed JSONs) and, for every 9 cells + mean9 (mean_per_cell), computes:
armA mean+std, armB mean+std, delta mean+std, and fraction of seeds delta>0.
Writes AB_MULTISEED.md. No DB, no kNN, no re-enrichment.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path("/home/frapercan/Thesis2/storage/regen_headline/protst_ab")
SEEDS = [42, 7, 123, 2024]
CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco",
         "pk-mfo", "pk-bpo", "pk-cco"]
FOCUS = ["lk-bpo", "pk-bpo", "nk-bpo", "nk-mfo", "nk-cco", "lk-mfo"]


def paths(seed, arm):
    if seed == 42:
        return HERE / f"{arm}_9cell.json"
    return HERE / f"{arm}_seed{seed}_9cell.json"


def load(seed, arm):
    d = json.loads(paths(seed, arm).read_text())["reranked"]
    out = {c: d["per_cell"][c]["f_micro_w"] for c in CELLS}
    out["mean9"] = d["mean_per_cell"]
    return out


def main():
    A = {s: load(s, "armA") for s in SEEDS}
    B = {s: load(s, "armB") for s in SEEDS}
    keys = CELLS + ["mean9"]

    rows = {}
    for k in keys:
        a = np.array([A[s][k] for s in SEEDS], dtype=float)
        b = np.array([B[s][k] for s in SEEDS], dtype=float)
        dl = b - a
        rows[k] = dict(
            a_mean=a.mean(), a_std=a.std(ddof=0),
            b_mean=b.mean(), b_std=b.std(ddof=0),
            d_mean=dl.mean(), d_std=dl.std(ddof=0),
            pos_frac=float((dl > 0).mean()), deltas=dl.tolist(),
        )

    def fmt(k):
        r = rows[k]
        return (f"| {k:<7} | {r['a_mean']:.4f} +- {r['a_std']:.4f} "
                f"| {r['b_mean']:.4f} +- {r['b_std']:.4f} "
                f"| {r['d_mean']:+.4f} +- {r['d_std']:.4f} "
                f"| {int(r['pos_frac']*len(SEEDS))}/{len(SEEDS)} |")

    hdr = ("| cell    | armA (base)        | armB (+protst)     "
           "| delta (B-A)         | pos/seeds |\n"
           "|---------|--------------------|--------------------|"
           "---------------------|-----------|")

    focus_tbl = hdr + "\n" + "\n".join(fmt(k) for k in FOCUS + ["mean9"])
    all_tbl = hdr + "\n" + "\n".join(fmt(k) for k in keys)

    # per-seed raw delta table for the focus cells
    seed_hdr = "| cell    | " + " | ".join(f"s{ s}" for s in SEEDS) + " |\n"
    seed_hdr += "|---------|" + "|".join(["--------"] * len(SEEDS)) + "|"
    seed_rows = []
    for k in FOCUS + ["mean9"]:
        vals = " | ".join(f"{d:+.4f}" for d in rows[k]["deltas"])
        seed_rows.append(f"| {k:<7} | {vals} |")
    seed_tbl = seed_hdr + "\n" + "\n".join(seed_rows)

    # verdict logic
    lkbpo, pkbpo = rows["lk-bpo"], rows["pk-bpo"]
    nkbpo = rows["nk-bpo"]
    mean9 = rows["mean9"]

    def stable_pos(r):
        return r["pos_frac"] == 1.0 and r["d_mean"] > r["d_std"]

    def sign_consistent(r):
        # all deltas same sign as the mean
        return all((d > 0) == (r["d_mean"] > 0) for d in r["deltas"])

    reg_cells = ["nk-mfo", "nk-cco", "lk-mfo"]
    reg_lines = []
    for k in reg_cells:
        r = rows[k]
        noise = abs(r["d_mean"]) < r["d_std"]
        allneg = all(d < 0 for d in r["deltas"])
        verdict = ("within seed std (likely NOISE)" if noise
                   else "|mean| > std" + (" AND all 4 seeds negative (REAL dilution)"
                                          if allneg else " but sign flips across seeds"))
        reg_lines.append(f"- **{k}**: delta {r['d_mean']:+.4f} +- {r['d_std']:.4f}, "
                         f"{int(r['pos_frac']*4)}/4 positive -> {verdict}")

    md = f"""# ProtST reranker A/B - multi-seed robustness (sealed 227->230 frame)

4 seeds {{42, 7, 123, 2024}} of the SAME offline A/B: armA = champion baseline
(3 protst columns excluded), armB = champion + protst (columns kept). Same enriched
parquets (`enriched_train.parquet` / `enriched_eval.parquet`, NO re-enrichment), same
harness (`train_ab_seed.py`, `score_ab.py`), same PARAMS. The ONLY per-run change is the
LightGBM master `seed`, which cascades to bagging_seed / feature_fraction_seed /
data_random_seed (empirically verified: same seed -> identical predictions, different seed
-> divergent). Seed 42 reuses the existing `armA_9cell.json` / `armB_9cell.json`.

## Per-cell across-seed table (focus cells + mean9)

std is population std (ddof=0) over the 4 seeds; delta = armB - armA; pos/seeds = seeds with delta>0.

{focus_tbl}

## Per-seed delta (armB - armA), focus cells

{seed_tbl}

## All 9 cells + mean9

{all_tbl}

## Verdict

**(a) BP-wall lifts stable + larger than seed noise?**
- **lk-bpo**: delta {lkbpo['d_mean']:+.4f} +- {lkbpo['d_std']:.4f}, {int(lkbpo['pos_frac']*4)}/4 seeds positive; |mean| {'>' if lkbpo['d_mean']>lkbpo['d_std'] else '<='} std -> {'STABLE POSITIVE, exceeds seed noise' if stable_pos(lkbpo) else 'positive but not clearly above noise'}.
- **pk-bpo**: delta {pkbpo['d_mean']:+.4f} +- {pkbpo['d_std']:.4f}, {int(pkbpo['pos_frac']*4)}/4 seeds positive; |mean| {'>' if pkbpo['d_mean']>pkbpo['d_std'] else '<='} std -> {'STABLE POSITIVE, exceeds seed noise' if stable_pos(pkbpo) else 'positive but not clearly above noise'}.
- **nk-bpo**: delta {nkbpo['d_mean']:+.4f} +- {nkbpo['d_std']:.4f}, {int(nkbpo['pos_frac']*4)}/4 seeds positive; |mean| {'>' if nkbpo['d_mean']>nkbpo['d_std'] else '<='} std -> {'STABLE POSITIVE' if stable_pos(nkbpo) else 'positive but not clearly above noise'}.
- Sign-consistency: lk-bpo {'all-positive' if sign_consistent(lkbpo) and lkbpo['d_mean']>0 else 'sign flips'}, pk-bpo {'all-positive' if sign_consistent(pkbpo) and pkbpo['d_mean']>0 else 'sign flips'}, nk-bpo {'all-positive' if sign_consistent(nkbpo) and nkbpo['d_mean']>0 else 'sign flips'}.

**(b) Are the MF/CC regressions noise or real dilution?**
{chr(10).join(reg_lines)}

**(c) mean9 delta stable-positive?**
- mean9 delta {mean9['d_mean']:+.4f} +- {mean9['d_std']:.4f}, {int(mean9['pos_frac']*4)}/4 seeds positive; |mean| {'>' if mean9['d_mean']>mean9['d_std'] else '<='} std -> {'STABLE NET POSITIVE' if stable_pos(mean9) else 'net positive on average but within seed noise'}.

## Non-committal notes

- This is an OFFLINE A/B delta vs OUR champion, not the platform eval and not the leaderboard
  gap. Nothing here promotes or touches the served reranker.
- Single seed knob (`seed`) controls all LightGBM RNG (verified); no separate bagging /
  feature-fraction knob had to be set, so all 4 seeds sit on identical footing.
"""
    (HERE / "AB_MULTISEED.md").write_text(md)
    print(md)
    print("\nWROTE", HERE / "AB_MULTISEED.md")


if __name__ == "__main__":
    main()
