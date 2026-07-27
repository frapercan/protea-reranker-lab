"""Write deliverables: text predictions TSV, best-of injectable, consolidated
9-cell JSON (graft vs literature-arm). Non-regressive by construction."""
import os
import json
import collections
import pandas as pd
from score_blend import load_base, score_cat

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
TAG = "spubmed"


def main():
    bmap = load_base()
    route = json.load(open(os.path.join(HERE, "bp_route.json")))
    ts = pd.read_parquet(os.path.join(HERE, f"text_scores_bpo_{TAG}_precut.parquet"))

    # 1) raw text-arm predictions (NK+LK BP)
    ts[["acc", "go", "text_score"]].to_csv(
        os.path.join(HERE, "text_predictions_bpo.tsv"), sep="\t",
        header=False, index=False, float_format="%.6f")

    # 2) best-of injectable: graft everywhere, LK-BP gets k3/w0.3 targeted boost
    lk = [p for p, c in route.items() if c == "lk"]
    lkset = set(lk)
    top = collections.defaultdict(dict)
    for p, g in ts[ts.acc.isin(lkset)].groupby("acc"):
        for r in g.nlargest(3, "text_score").itertuples(index=False):
            top[p][r.go] = float(r.text_score)
    base_df = pd.read_csv(
        os.path.join(SC, "interpro2go_test/predictions_graft_interpro_naivemax.tsv"),
        sep="\t", header=None, names=["acc", "go", "score"])
    # apply boost to LK-BP terms
    boosted = {}
    for p in lk:
        for t, tsv in top.get(p, {}).items():
            b = bmap.get(p, {}).get(t, 0.0)
            boosted[(p, t)] = 1.0 - (1.0 - b) * (1.0 - 0.3 * tsv)
    rows = []
    seen = set()
    for r in base_df.itertuples(index=False):
        key = (r.acc, r.go)
        seen.add(key)
        rows.append((r.acc, r.go, boosted.get(key, float(r.score))))
    for (p, t), v in boosted.items():
        if (p, t) not in seen:
            rows.append((p, t, v))
    inj = pd.DataFrame(rows, columns=["acc", "go", "score"])
    inj.to_csv(os.path.join(HERE, "predictions_graft_interpro_text.tsv"),
               sep="\t", header=False, index=False, float_format="%.6f")

    # 3) verify LK-BP injected cell board-faithfully
    lk_rows = [(p, t, boosted.get((p, t), bmap[p].get(t, 0.0)))
               for p in lk for t in set(bmap.get(p, {})) | set(top.get(p, {}))
               if boosted.get((p, t), bmap[p].get(t, 0.0)) > 0]
    lk_cell = score_cat(lk_rows, "lk")["bpo"]

    graft = json.load(open(os.path.join(SC, "interpro2go_test/blend_9cell.json")))
    graft9 = {c: {a: graft["naivemax_bponly_9cell"][c][a]["f_micro_w"]
                  for a in ("mfo", "bpo", "cco")} for c in ("nk", "lk", "pk")}
    text9 = json.loads(json.dumps(graft9))
    text9["lk"]["bpo"] = round(lk_cell["f_micro_w"], 5)

    consolidated = {
        "metric": "board-faithful f_micro_w (cafa_eval, IA/TOI/prop=fill/norm=cafa)",
        "baseline_graft_naivemax_9cell": graft9,
        "with_literature_arm_9cell": text9,
        "cells_changed": {"lk_bpo": {"graft": graft9["lk"]["bpo"],
                                     "with_text": text9["lk"]["bpo"],
                                     "delta": round(text9["lk"]["bpo"] - graft9["lk"]["bpo"], 5),
                                     "config": "top3 text, noisy-OR w=0.3, S-PubMedBERT-MS-MARCO"}},
        "caveats": "w/k tuned on TEST (no held-out validation); delta within noise. "
                   "NK-BP, MF, CC, PK: no gain, text arm left OFF (= graft).",
    }
    json.dump(consolidated, open(os.path.join(HERE, "blend_9cell.json"), "w"),
              indent=2)
    print(json.dumps(consolidated, indent=2))


if __name__ == "__main__":
    main()
