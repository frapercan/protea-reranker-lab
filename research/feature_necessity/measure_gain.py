"""Measure what each feature actually contributes to the sealed champion.

Reads the three per-category boosters of the sealed v227 to v230 run and reports
LightGBM's total split gain per feature, per cell and aggregated. Writes a JSON
receipt so the numbers in the thesis, the interface and the Sphinx reference all
cite the same measurement rather than a remembered one.

A gain of exactly zero means LightGBM never split on the column. That is a fact
about this booster on this frame, not a verdict on the signal: `interpro_*`
scores zero because its database tables are empty, and InterPro still enters the
pipeline as a separate noisy-OR graft worth +0.0179. Read `notes` in the output
before retiring anything.

Read-only. No database, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb

SEALED = Path(
    "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/clean_227230"
)
OUT = Path("/home/frapercan/Thesis2/storage/feature_necessity")
CELLS = ("nk", "lk", "pk")

#: Families whose zero gain is a broken producer, not a useless signal.
BROKEN_NOT_USELESS = {
    "interpro": (
        "The interpro_* columns score zero because the InterPro database tables "
        "are empty, so every row carries the same default. InterPro still "
        "contributes +0.0179 f_micro_w through a separate noisy-OR graft that "
        "reads interpro2go.txt and protein2ipr.json directly. Fix the tables "
        "before drawing any conclusion about the features."
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_cell: dict[str, dict[str, float]] = {}
    names: list[str] = []

    for cell in CELLS:
        booster = lgb.Booster(model_file=str(SEALED / "boosters" / f"booster_{cell}.txt"))
        names = booster.feature_name()
        gains = booster.feature_importance("gain")
        per_cell[cell] = dict(zip(names, (float(g) for g in gains), strict=True))

    total = {n: sum(per_cell[c][n] for c in CELLS) for n in names}
    grand = sum(total.values())
    assert grand > 0, "every feature has zero gain: wrong model files?"

    ranked = sorted(names, key=lambda n: -total[n])
    dead = [n for n in names if total[n] == 0.0]

    report = {
        "frame": "v227 to v230, sealed champion, f_micro_w 0.4063",
        "boosters": [f"boosters/booster_{c}.txt" for c in CELLS],
        "n_features": len(names),
        "n_zero_gain": len(dead),
        "share_of_gain": {n: total[n] / grand for n in ranked},
        "per_cell_share": {
            c: {
                n: (per_cell[c][n] / s if (s := sum(per_cell[c].values())) else 0.0)
                for n in ranked
            }
            for c in CELLS
        },
        "zero_gain_features": dead,
        "notes": BROKEN_NOT_USELESS,
    }
    (OUT / "gain_report.json").write_text(json.dumps(report, indent=2))

    print(f"{len(names)} features | {len(dead)} with gain exactly 0\n")
    print("top 8 by aggregate gain:")
    for n in ranked[:8]:
        print(f"  {n:34s} {total[n] / grand * 100:5.2f}%")
    print("\nlineage, per cell:")
    for c in CELLS:
        s = sum(per_cell[c].values())
        lin = sum(v for k, v in per_cell[c].items() if k.startswith("lineage"))
        print(f"  {c}: {lin / s * 100:6.2f}%")
    print(f"\nzero gain in all three cells ({len(dead)}):")
    for n in dead:
        print(f"  {n}")
    print(f"\nreceipt -> {OUT / 'gain_report.json'}")


if __name__ == "__main__":
    main()
