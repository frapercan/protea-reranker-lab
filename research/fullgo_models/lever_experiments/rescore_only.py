"""Re-score an already-applied lever (pred_pk.tsv / gt_pk.tsv exist) without
re-running the booster apply. Uses apply_score_pk.score_pk (absolute-path fix).
"""
import argparse
import json
from pathlib import Path

from apply_score_pk import score_pk

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

v, per_ns = score_pk(args.out)
out = {"tag": args.tag, "pk_f_micro_w": round(v, 6),
       "per_ns": {k: round(x, 6) for k, x in per_ns.items()},
       "baseline": 0.4142, "delta": round(v - 0.4142, 6)}
with open(Path(args.out) / f"result_{args.tag}.json", "w") as w:
    json.dump(out, w, indent=2)
print(f"PK f_micro_w = {v:.4f}  delta={v-0.4142:+.4f}")
print(json.dumps(out))
