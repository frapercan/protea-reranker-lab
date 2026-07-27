"""Score the three arms in the TRUE board frame + paired bootstrap CI over proteins.

For each cell (LK-BPO, PK-BPO) and arm (deployed, plainIA, freqpart):
  - authoritative BP f_micro_w + best tau via cafa_eval
  - per-protein weighted confusion at that arm's best tau (bootstrap engine, validated vs cafaeval)
Paired bootstrap: proteins are the SAME set across arms (shared gt); resample protein indices, recompute
each arm's f_micro_w at its FIXED full-data best tau, delta = f_arm - f_deployed.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/frapercan/Thesis2/storage/transfew_calib")
import frame

W = "/home/frapercan/Thesis2/storage/transfew_calib"
ARMS = ("deployed", "plainIA", "freqpart")
N_BOOT = 2000
SEED = 42
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


def boot_delta(conf_dep, conf_arm, n_boot, seed):
    """Paired bootstrap of (f_arm - f_dep) over proteins. conf_* = (wtp,wfp,wfn) arrays, aligned."""
    wtp_d, wfp_d, wfn_d = conf_dep
    wtp_a, wfp_a, wfn_a = conf_arm
    P = len(wtp_d)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, P, P)
        def f(wtp, wfp, wfn):
            TP = wtp[idx].sum(); FP = wfp[idx].sum(); FN = wfn[idx].sum()
            if TP <= 0:
                return 0.0
            pr = TP / (TP + FP); rc = TP / (TP + FN)
            return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
        deltas[b] = f(wtp_a, wfp_a, wfn_a) - f(wtp_d, wfp_d, wfn_d)
    return deltas


def main():
    result = {"frame": frame.GT and "TRUE board (prop=fill norm=cafa no_orphans toi; PK -known)",
              "metric": "f_micro_w on biological_process, max over tau", "n_boot": N_BOOT, "cells": {}}
    for cat in ("lk", "pk"):
        log(f"==== {cat.upper()}-BPO ====")
        pts = {}
        confs = {}
        for arm in ARMS:
            d = f"{W}/pred_{cat}_{arm}"
            pt = frame.score_dir(d, cat, n_cpu=8)
            pts[arm] = pt
            M = frame.build_bp_matrices(f"{d}/{cat}.tsv", cat, n_cpu=8)
            wtp, wfp, wfn = frame.per_protein_confusion(M, pt["tau"])
            confs[arm] = (wtp, wfp, wfn)
            # sanity: summed micro-f at this tau must match cafaeval point
            f_chk = frame.micro_f_from_confusion(wtp, wfp, wfn)
            log(f"  {arm:9s} f_micro_w={pt['f_micro_w']:.5f} tau={pt['tau']:.2f} "
                f"(engine {f_chk:.5f}) P={len(wtp)}")
        cell = {"n_proteins": int(len(confs['deployed'][0])),
                "deployed_f": pts["deployed"]["f_micro_w"], "deployed_tau": pts["deployed"]["tau"],
                "arms": {}}
        for arm in ARMS:
            delta_pt = pts[arm]["f_micro_w"] - pts["deployed"]["f_micro_w"]
            db = boot_delta(confs["deployed"], confs[arm], N_BOOT, SEED)
            cell["arms"][arm] = {
                "f_micro_w": pts[arm]["f_micro_w"], "tau": pts[arm]["tau"],
                "pr_micro_w": pts[arm]["pr_micro_w"], "rc_micro_w": pts[arm]["rc_micro_w"],
                "cov": pts[arm]["cov"],
                "delta_vs_deployed": round(delta_pt, 5),
                "boot_mean_delta": round(float(db.mean()), 5),
                "boot_ci95": [round(float(np.percentile(db, 2.5)), 5),
                              round(float(np.percentile(db, 97.5)), 5)],
                "boot_p_gt_0": round(float((db > 0).mean()), 4),
            }
            log(f"  DELTA {arm:9s} {delta_pt:+.5f}  boot95=["
                f"{np.percentile(db,2.5):+.5f},{np.percentile(db,97.5):+.5f}]  P(>0)={(db>0).mean():.3f}")
        result["cells"][f"{cat}_bpo"] = cell
        json.dump(result, open(f"{W}/score_arms.json", "w"), indent=2)
    log("score_arms.json written")


if __name__ == "__main__":
    main()
