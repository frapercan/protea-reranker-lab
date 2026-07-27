"""Score the native-booster validation-frame predictions with cafaeval, reproducing
the offline-seal recipe as faithfully as the available frame artefacts allow.

Recipe (matches ensemble_seal.py, the offline-champion driver):
    cafa_eval(OBO, pred_dir, gt, ia=IA, prop="fill", norm="cafa",
              no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1 [, toi_file][, exclude])
    f_micro_w per category = mean over the 3 namespaces of the per-ns MAX f_micro_w.

Frame-specific knobs (toi_file, PK exclude) are TEST-frame (227->230) CAFA artefacts
that do NOT exist for the validation frame (220->227). We therefore run WITHOUT them
as the primary number (documented as a caveat), and optionally WITH a proxy TOI.

Uses the cafaeval-protea fork (PROTEA .venv) which supports toi_file/exclude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cafaeval.evaluation import cafa_eval

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
HERE = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_v5/validation_score"
CATS = ("nk", "lk", "pk")


def score_one(cat: str, toi_file: str | None, exclude: str | None) -> float:
    pred = Path(HERE) / f"pred_{cat}.tsv"
    gt = Path(HERE) / f"gt_{cat}.tsv"
    # cafaeval discovers prediction files under a directory; isolate this one.
    pdir = Path(HERE) / f"_pred_dir_{cat}"
    pdir.mkdir(exist_ok=True)
    link = pdir / "m.tsv"
    if link.exists():
        link.unlink()
    link.symlink_to(pred)
    kw = dict(ia=IA, prop="fill", norm="cafa", no_orphans=True, max_terms=None,
              th_step=0.01, n_cpu=1)
    if toi_file:
        kw["toi_file"] = toi_file
    if exclude:
        kw["exclude"] = exclude
    df, _ = cafa_eval(OBO, str(pdir), str(gt), **kw)
    sub = df.reset_index()
    col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
    per_ns = sub.groupby("ns")[col].max()
    return float(per_ns.mean()), {ns: float(v) for ns, v in per_ns.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toi", default=None, help="proxy TOI file (optional)")
    ap.add_argument("--pk-exclude", default=None, help="PK known-exclude file (optional)")
    ap.add_argument("--tag", default="no_toi")
    args = ap.parse_args()

    res = {}
    perns = {}
    for cat in CATS:
        excl = args.pk_exclude if cat == "pk" else None
        v, ns = score_one(cat, args.toi, excl)
        res[cat] = v
        perns[cat] = ns
        print(f"{cat.upper()}: f_micro_w = {v:.4f}   per-ns={ {k: round(x,4) for k,x in ns.items()} }", flush=True)
    mean = sum(res.values()) / 3
    print(f"MEAN: {mean:.4f}   (tag={args.tag})", flush=True)
    out = {
        "tag": args.tag, "toi": args.toi, "pk_exclude": args.pk_exclude,
        "f_micro_w": {c: round(res[c], 6) for c in CATS},
        "f_micro_w_per_ns": {c: {k: round(v, 6) for k, v in perns[c].items()} for c in CATS},
        "mean": round(mean, 6),
    }
    with open(Path(HERE) / f"result_{args.tag}.json", "w") as w:
        json.dump(out, w, indent=2)


if __name__ == "__main__":
    main()
