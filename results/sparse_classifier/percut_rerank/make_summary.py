"""Assemble SUMMARY.md and the final result_9cell.json (with board deltas)."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))

BOARD = {  # current live board: PROTEA-reranked (#1 in 5/9)
    "nk": {"mfo": 0.602, "bpo": 0.309, "cco": 0.431},
    "lk": {"mfo": 0.519, "bpo": 0.348, "cco": 0.419},
    "pk": {"mfo": 0.235, "bpo": 0.117, "cco": 0.254},
}
ASPECTS = ["mfo", "bpo", "cco"]
CATS = ["nk", "lk", "pk"]


def main():
    with open(os.path.join(HERE, "result_9cell.json")) as f:
        res = json.load(f)
    # idempotent: if already enriched, recover the raw per-cell detail
    if "detail" in res:
        res = res["detail"]
    vproxy = {}
    vp = os.path.join(HERE, "validation_proxy_9cell.json")
    if os.path.exists(vp):
        with open(vp) as f:
            vproxy = json.load(f)

    test = {c: {a: (res[c][a]["f_micro_w"] if res[c].get(a) else None) for a in ASPECTS} for c in CATS}

    # deltas
    deltas = {c: {a: (round(test[c][a] - BOARD[c][a], 4) if test[c][a] is not None else None)
                  for a in ASPECTS} for c in CATS}

    improved = []
    regressed = []
    all_test, all_board = [], []
    for c in CATS:
        for a in ASPECTS:
            t = test[c][a]
            if t is None:
                continue
            all_test.append(t); all_board.append(BOARD[c][a])
            if t - BOARD[c][a] > 0:
                improved.append((c, a, round(t - BOARD[c][a], 4)))
            elif t - BOARD[c][a] < 0:
                regressed.append((c, a, round(t - BOARD[c][a], 4)))
    mean_test = sum(all_test) / len(all_test)
    mean_board = sum(all_board) / len(all_board)

    # write enriched result json
    enriched = {"test_f_micro_w": test, "board": BOARD, "delta_vs_board": deltas,
                "mean_test": round(mean_test, 4), "mean_board": round(mean_board, 4),
                "validation_proxy_f_micro_w": vproxy,
                "cells_improved": len(improved), "cells_total": len(all_test),
                "detail": res}
    with open(os.path.join(HERE, "result_9cell.json"), "w") as f:
        json.dump(enriched, f, indent=2)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "

    def dfmt(x):
        if x is None:
            return "  -   "
        s = f"{x:+.3f}"
        return s

    lines = []
    lines.append("# Per-cut sparse-classifier reranker: 9-cell LAFA result\n")
    lines.append("Per-category LightGBM reranker over the frozen "
                 "`clean-learned-sparseclf-assoc-train227-test230` dataset. "
                 "TEST = v227-v230 LAFA frame, scored with cafaeval "
                 "(`-ia IA.tsv -prop fill -norm cafa -no_orphans -toi <toi>`, "
                 "`-known PK_known` for PK). Metric = f_micro_w (IA-weighted micro-Fmax).\n")

    lines.append("## TEST 9-cell f_micro_w (vs current live board)\n")
    lines.append("| cell | reranker | board | delta |")
    lines.append("|------|---------:|------:|------:|")
    for c in CATS:
        for a in ASPECTS:
            lines.append(f"| {c.upper()}-{a} | {fmt(test[c][a])} | {fmt(BOARD[c][a])} | {dfmt(deltas[c][a])} |")
    lines.append(f"| **MEAN** | **{mean_test:.3f}** | **{mean_board:.3f}** | **{mean_test-mean_board:+.3f}** |\n")

    lines.append(f"- Cells improved vs board: **{len(improved)}/{len(all_test)}**.")
    if improved:
        lines.append("  - " + ", ".join(f"{c.upper()}-{a} ({d:+.3f})" for c, a, d in sorted(improved, key=lambda x: -x[2])))
    if regressed:
        lines.append(f"- Cells regressed: {len(regressed)}.")
        lines.append("  - " + ", ".join(f"{c.upper()}-{a} ({d:+.3f})" for c, a, d in sorted(regressed, key=lambda x: x[2])))
    lines.append("")

    lines.append("## VALIDATION (v225-v227) selection proxy 9-cell\n")
    lines.append("Flat IA-weighted micro-Fmax on the validation candidate pool "
                 "(NOT cafaeval-propagated; used only for early-stopping / lever "
                 "confirmation, never for tuning on TEST).\n")
    if vproxy:
        lines.append("| cell | valid proxy f |")
        lines.append("|------|--------------:|")
        for c in CATS:
            for a in ASPECTS:
                v = vproxy.get(c, {}).get(a)
                lines.append(f"| {c.upper()}-{a} | {fmt(v)} |")
        lines.append("")

    lines.append("## Protocol\n")
    lines.append("- Per-category LightGBM binary classifier (objective=binary, "
                 "early stopping on the v225-v227 hold-out by AUC). `aspect` is a "
                 "feature; models are NOT split per aspect.")
    lines.append("- TRAIN = snapshot_pairs v160-v165 .. v220-v225; VALID (early "
                 "stop + selection) = v225-v227; TEST = v227-v230 (eval.parquet).")
    lines.append("- Candidate pool: NK/LK use the full pool (knn+classifier+"
                 "association); PK restricted to knn_present candidates "
                 "(classifier/association dilute PK precision), applied "
                 "consistently across train/valid/test.")
    lines.append("- Lever (keeper): `pminmax` on LK-bpo only = per-protein "
                 "min-max of the reranker score before thresholding.")
    lines.append("- Features: 69 numeric+bool columns + encoded `aspect_code` "
                 "(70 total; the full frozen schema beyond the documented core "
                 "set was used).\n")

    lines.append("## Caveats / footnotes\n")
    lines.append("1. **Mixed-KNN-backend provenance (minor):** train snapshot_pairs "
                 "v190-v195 and v195-v200 were computed with exact numpy KNN; all "
                 "other train pairs, the v225-v227 validation, and the v227-v230 TEST "
                 "use faiss IVFFlat (approximate). TEST and VALIDATION are therefore "
                 "backend-consistent; the mix only adds slight noise to two early "
                 "train splits.")
    lines.append("2. This is the **per-cut** sparse-classifier result: co-annotation / "
                 "classifier / association features are computed temporally-honest "
                 "per cut (no future leakage), which is the point of the experiment.")
    lines.append("3. The validation 9-cell are a flat candidate-pool proxy (no GO "
                 "propagation, no PK-known exclusion for v225-v227), so they are not "
                 "directly comparable in level to the cafaeval TEST cells; they are a "
                 "monotone selection signal only.\n")

    with open(os.path.join(HERE, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote SUMMARY.md and enriched result_9cell.json")
    print(f"mean_test={mean_test:.4f} mean_board={mean_board:.4f} improved={len(improved)}/{len(all_test)}")


if __name__ == "__main__":
    main()
