"""Blend sweep: final = (1-w)*base + w*feat, per BP candidate. Score TEST via cafa_eval.

Also runs the lab's validation IA-weighted micro-Fmax proxy to SELECT w without
touching test, then reports the validation-selected w's test cell vs the oracle sweep.
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from score_bp import score_bp_cell

HERE = os.path.dirname(os.path.abspath(__file__))
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def ia_micro_fmax(df, score_col, ia, th_step=0.01):
    """Flat IA-weighted micro-Fmax proxy (matches train_rerank selection proxy)."""
    w = df["go_term_id"].map(ia).fillna(0.0).values
    s = df[score_col].values
    y = df["label"].values.astype(bool)
    wy = w * y; wny = w * (~y); tot_pos = wy.sum()
    best, bt = 0.0, 0.0
    for tau in np.arange(0.0, 1.0 + 1e-9, th_step):
        sel = s >= tau
        tp = wy[sel].sum(); fp = wny[sel].sum(); fn = tot_pos - tp
        if tp <= 0:
            continue
        pr = tp / (tp + fp); rc = tp / (tp + fn)
        if pr + rc > 0:
            f = 2 * pr * rc / (pr + rc)
            if f > best:
                best, bt = f, tau
    return float(best), float(bt)


def blend(df, w, feat):
    return (1 - w) * df["base_score"].values + w * df[feat].values


def main():
    ia = C.load_ia()
    feat = sys.argv[1] if len(sys.argv) > 1 else "profile_fit"
    results = {"feature": feat, "cats": {}}

    for cat in ["lk", "pk"]:
        print(f"\n########## {cat.upper()} feat={feat} ##########")
        test = pd.read_parquet(os.path.join(HERE, f"test_{cat}_bp_feat.parquet"))
        valid = pd.read_parquet(os.path.join(HERE, f"valid_{cat}_bp_feat.parquet"))

        # --- validation proxy selection ---
        vsel = []
        for w in WS:
            valid["_b"] = blend(valid, w, feat)
            f, t = ia_micro_fmax(valid, "_b", ia)
            vsel.append((w, f, t))
        best_v = max(vsel, key=lambda r: r[1])
        print("valid proxy:", [(round(w,1), round(f,4)) for w, f, _ in vsel])
        print("valid-selected w =", best_v[0], "proxy_fmax=", round(best_v[1], 4))

        # --- test sweep (oracle diagnostic) ---
        sweep = {}
        for w in WS:
            t0 = time.time()
            test["_b"] = blend(test, w, feat)
            cell = score_bp_cell(cat, test, "_b")
            sweep[f"{w:.1f}"] = cell
            print(f"  w={w:.1f} TEST bpo f={cell['f_micro_w']:.4f} tau={cell['tau']:.2f} "
                  f"pr={cell['pr_micro_w']:.3f} rc={cell['rc_micro_w']:.3f} ({time.time()-t0:.0f}s)")
        best_test_w = max(sweep, key=lambda k: sweep[k]["f_micro_w"])
        results["cats"][cat] = {
            "valid_selection": {"w": best_v[0], "proxy_fmax": best_v[1],
                                 "proxy_curve": [(w, f) for w, f, _ in vsel]},
            "valid_selected_test_cell": sweep[f"{best_v[0]:.1f}"],
            "test_sweep": sweep,
            "oracle_test_w": float(best_test_w),
            "oracle_test_cell": sweep[best_test_w],
            "baseline_cell": sweep["0.0"],
        }

    out = os.path.join(HERE, f"blend_{feat}.json")
    json.dump(results, open(out, "w"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
