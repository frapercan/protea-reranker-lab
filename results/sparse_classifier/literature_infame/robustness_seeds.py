"""Seed-robustness: confirm the literature BP gain sign is stable, not seed luck.
Retrain base vs lit at alternate seeds, score board-faithful, record deltas."""
import os, json
import pandas as pd
import train_rerank_lit as T
import graft_score as G

HERE = os.path.dirname(os.path.abspath(__file__))


def score(cat, bp):
    u = G.union_naivemax(bp)
    return G.score_bp(list(zip(u.acc, u.go, u.s_max)), cat)["f_micro_w"]


def main():
    text_df = pd.read_parquet(T.TEXT_SCORES)
    plan = {"lk": [7, 123, 2024], "pk": [7]}
    out = {}
    for cat, seeds in plan.items():
        out[cat] = []
        for s in seeds:
            T.SEED = s
            bp_base, _, _ = T.train_variant(cat, False, text_df)
            bp_lit, _, _ = T.train_variant(cat, True, text_df)
            fb, fl = score(cat, bp_base), score(cat, bp_lit)
            rec = {"seed": s, "base": fb, "lit": fl, "delta": round(fl - fb, 5)}
            out[cat].append(rec)
            print(cat, rec, flush=True)
    json.dump(out, open(os.path.join(HERE, "robustness_seeds.json"), "w"), indent=2)
    print("written robustness_seeds.json", flush=True)


if __name__ == "__main__":
    main()
