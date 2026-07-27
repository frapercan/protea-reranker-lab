"""2-seed directional robustness aggregation for the ProtST reranker A/B.

Seeds {42, 7}. seed42 = existing armA_9cell.json / armB_9cell.json; seed7 = new runs.
For each cell: armA/armB per seed, delta (armB-armA) per seed, and sign-consistency
across the two seeds. Writes AB_MULTISEED.md. No DB, no kNN, no re-enrichment.
"""
import json
from pathlib import Path

HERE = Path("/home/frapercan/Thesis2/storage/regen_headline/protst_ab")
CELLS = ["nk-mfo", "nk-bpo", "nk-cco", "lk-mfo", "lk-bpo", "lk-cco",
         "pk-mfo", "pk-bpo", "pk-cco"]
BP = ["nk-bpo", "lk-bpo", "pk-bpo"]
MFCC_REG = ["nk-mfo", "nk-cco", "lk-mfo"]


def path(seed, arm):
    return HERE / (f"{arm}_9cell.json" if seed == 42 else f"{arm}_seed{seed}_9cell.json")


def load(seed, arm):
    d = json.loads(path(seed, arm).read_text())["reranked"]
    out = {c: d["per_cell"][c]["f_micro_w"] for c in CELLS}
    out["mean9"] = d["mean_per_cell"]
    return out


A42, B42 = load(42, "armA"), load(42, "armB")
A7, B7 = load(7, "armA"), load(7, "armB")
keys = CELLS + ["mean9"]

d42 = {k: B42[k] - A42[k] for k in keys}
d7 = {k: B7[k] - A7[k] for k in keys}


def sign_flag(k):
    a, b = d42[k], d7[k]
    if (a > 0) and (b > 0):
        return "both +"
    if (a < 0) and (b < 0):
        return "both -"
    return "FLIP"


rows = []
for k in keys:
    rows.append(
        f"| {k:<7} | {A42[k]:.4f} | {B42[k]:.4f} | {d42[k]:+.4f} "
        f"| {A7[k]:.4f} | {B7[k]:.4f} | {d7[k]:+.4f} | {sign_flag(k)} |"
    )

hdr = ("| cell    | s42 armA | s42 armB | s42 delta | s7 armA | s7 armB | s7 delta | sign |\n"
       "|---------|----------|----------|-----------|---------|---------|----------|------|")
table = hdr + "\n" + "\n".join(rows)

bp_both_pos = all(sign_flag(k) == "both +" for k in BP)
reg_both_neg = {k: sign_flag(k) == "both -" for k in MFCC_REG}
reg_flip = [k for k in MFCC_REG if sign_flag(k) == "FLIP"]
mean9_flag = sign_flag("mean9")

bp_line = ("BP wall (nk/lk/pk-bpo): "
           + ("POSITIVE IN BOTH SEEDS for all three "
              f"(lk-bpo {d42['lk-bpo']:+.4f}/{d7['lk-bpo']:+.4f}, "
              f"pk-bpo {d42['pk-bpo']:+.4f}/{d7['pk-bpo']:+.4f}, "
              f"nk-bpo {d42['nk-bpo']:+.4f}/{d7['nk-bpo']:+.4f}); the BP lift is real and sign-stable."
              if bp_both_pos else
              "NOT positive in both seeds for all three; see table."))

if reg_flip and not any(reg_both_neg.values()):
    reg_verdict = ("MF/CC dips sign-FLIP across the two seeds ("
                   + ", ".join(f"{k} {d42[k]:+.4f}/{d7[k]:+.4f}" for k in MFCC_REG)
                   + "), i.e. likely SEED NOISE, not real dilution.")
elif all(reg_both_neg.values()):
    reg_verdict = ("MF/CC dips are negative in BOTH seeds for all three ("
                   + ", ".join(f"{k} {d42[k]:+.4f}/{d7[k]:+.4f}" for k in MFCC_REG)
                   + "), i.e. a consistent, likely REAL dilution.")
else:
    stable = [k for k in MFCC_REG if reg_both_neg[k]]
    noisy = [k for k in MFCC_REG if not reg_both_neg[k]]
    reg_verdict = ("MF/CC dips are MIXED: "
                   + ("consistently negative (likely real): " + ", ".join(
                       f"{k} {d42[k]:+.4f}/{d7[k]:+.4f}" for k in stable) + "; " if stable else "")
                   + ("sign-flips (likely noise): " + ", ".join(
                       f"{k} {d42[k]:+.4f}/{d7[k]:+.4f}" for k in noisy) if noisy else ""))

mean9_line = (f"mean9 delta: seed42 {d42['mean9']:+.4f}, seed7 {d7['mean9']:+.4f} ("
              + {"both +": "net positive in both seeds",
                 "both -": "net negative in both seeds",
                 "FLIP": "sign flips across seeds"}[mean9_flag] + ").")

md = f"""# ProtST reranker A/B - 2-seed directional robustness (sealed 227->230 frame)

Directional cross-check of the ProtST reranker A/B on TWO LightGBM seeds {{42, 7}}. armA =
champion baseline (3 protst columns excluded); armB = champion + protst. Same enriched
parquets (`enriched_train.parquet` / `enriched_eval.parquet`, NO re-enrichment), same harness
(`train_ab_seed.py`, `score_ab.py`), same PARAMS; the only per-run change is the LightGBM
master `seed`, which cascades to bagging_seed / feature_fraction_seed / data_random_seed
(verified: same seed -> identical predictions, different seed -> divergent). seed42 reuses the
existing `armA_9cell.json` / `armB_9cell.json`; seed7 is the new pair. This is a directional
seed check only; the rigorous significance evidence is the protein-level bootstrap CIs on the
BP cells (which exclude zero), not this table.

## 9-cell + mean9 f_micro_w, both seeds

delta = armB - armA. sign = agreement of the two per-seed deltas.

{table}

## Verdict

- (a) {bp_line}
- (b) {reg_verdict}
- (c) {mean9_line}

## Non-committal notes

- Offline A/B delta vs OUR champion only. Nothing here touches the served reranker, promotes,
  or runs the platform eval. Informs the global-vs-BP-gated injection decision, nothing more.
- A single seed knob controls all LightGBM RNG (verified), so both seeds sit on identical footing.
"""

(HERE / "AB_MULTISEED.md").write_text(md)
print(md)
print("WROTE", HERE / "AB_MULTISEED.md")
