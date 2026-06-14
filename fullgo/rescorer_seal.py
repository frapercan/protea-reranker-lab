"""SEAL the learned re-scorer on the official 7401 frame.

Train per-category GBM on ALL SELECT 220->227 candidates; apply to the 7401 frame;
score with the EXACT published harness (toi + exclude=pk_known + Sep_2025 OBO).
Features: composite score + KNN sub-features + term IA + log t0-pool frequency.
Compare to the canonical composite KNN: NK 0.412 / LK 0.394 / PK 0.165 / mean 0.324.
"""
import math
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
from cafaeval.evaluation import cafa_eval

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
REL = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026"
TOI = f"{REL}/groundtruth_terms_of_interest.txt"
PKK_TEST = f"{REL}/groundtruth_PK_known.tsv"

SEL_KNN = "/tmp/select_knn_composite.tsv"
SEL_GT = "/tmp/select_gt_{cat}.tsv"
SEL_POOL = "/tmp/v220_exp_aspect.tsv"

TEST_KNN = "/tmp/canon_composite.tsv"
TEST_GT = REL + "/groundtruth_{cat}.tsv"
TEST_FREQ = "/tmp/v227_exp_freq.tsv"

BASE = {"NK": 0.412, "LK": 0.394, "PK": 0.165}


def parse_parents(obo):
    parents = defaultdict(set); cur = None
    for line in open(obo):
        line = line.strip()
        if line == "[Term]": cur = None
        elif line.startswith("id: GO:"): cur = line[4:]
        elif line.startswith("is_a:") and cur: parents[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            pr = line.split()
            if len(pr) >= 3: parents[cur].add(pr[2])
    return parents


def anc(t, parents, cache):
    if t in cache: return cache[t]
    out = set(); st = list(parents.get(t, ()))
    while st:
        a = st.pop()
        if a in out: continue
        out.add(a); st.extend(parents.get(a, ()))
    cache[t] = out; return out


def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return 0.0


def load_knn(path):
    """per (protein,term) -> [score, distance, id_nw, id_sw, tax_dist, vote]; max-score row."""
    f = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 11 or not c[1].startswith("GO:"): continue
            p, t = c[0], c[1]; sc = fnum(c[2]); k = (p, t)
            if k not in f or sc > f[k][0]:
                f[k] = [sc, fnum(c[3]), fnum(c[7]), fnum(c[8]), fnum(c[9]), fnum(c[10])]
    by = defaultdict(list)
    for (p, t), ft in f.items():
        by[p].append((t, ft))
    return by


def freq_from_pool(path, two_col=False):
    fr = defaultdict(int); seen = set()
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if two_col:
            if len(c) >= 2: fr[c[0]] = int(c[1])
        else:
            if len(c) >= 2:
                k = (c[0], c[1])
                if k not in seen: seen.add(k); fr[c[1]] += 1
    return fr


def gtset(path, parents, cache):
    gt = defaultdict(set)
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if not c or c[0] == "EntryID" or len(c) < 2 or not c[1].startswith("GO:"): continue
        gt[c[0]].add(c[1]); gt[c[0]] |= anc(c[1], parents, cache)
    return gt


def build_rows(knn_by_p, gt, ia, freq):
    rows = []
    for p in gt:
        gtp = gt[p]
        for t, ft in knn_by_p.get(p, ()):
            feat = ft + [ia.get(t, 0.0), math.log1p(freq.get(t, 0))]
            rows.append((p, t, feat, 1 if t in gtp else 0))
    return rows


def main():
    parents = parse_parents(OBO); cache = {}
    ia = {}
    for line in open(IA):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2: ia[c[0]] = fnum(c[1])
    sel_knn = load_knn(SEL_KNN)
    test_knn = load_knn(TEST_KNN)
    sel_freq = freq_from_pool(SEL_POOL)
    test_freq = freq_from_pool(TEST_FREQ, two_col=True)

    results = {}
    for cat in ("NK", "LK", "PK"):
        sel_gt = gtset(SEL_GT.format(cat=cat), parents, cache)
        test_gt = gtset(TEST_GT.format(cat=cat), parents, cache)
        tr_rows = build_rows(sel_knn, sel_gt, ia, sel_freq)
        te_rows = build_rows(test_knn, test_gt, ia, test_freq)
        Xtr = np.array([r[2] for r in tr_rows], dtype=np.float32)
        ytr = np.array([r[3] for r in tr_rows], dtype=np.int32)
        Xte = np.array([r[2] for r in te_rows], dtype=np.float32)
        dtr = lgb.Dataset(Xtr, label=ytr)
        params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                      min_data_in_leaf=50, feature_fraction=0.9, verbose=-1)
        b = lgb.train(params, dtr, num_boost_round=200)
        pred = b.predict(Xte)
        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td) / "pred"; pdir.mkdir()
            with open(pdir / "m.tsv", "w") as w:
                for (p, t, _, _), sc in zip(te_rows, pred):
                    w.write(f"{p}\t{t}\t{sc:.6f}\n")
            kw = dict(ia=IA, prop="fill", norm="cafa", no_orphans=True, max_terms=None,
                      th_step=0.01, n_cpu=1, toi_file=TOI)
            if cat == "PK": kw["exclude"] = PKK_TEST
            df, _ = cafa_eval(OBO, str(pdir), TEST_GT.format(cat=cat), **kw)
            sub = df.reset_index()
            col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
            v = float(sub.groupby("ns")[col].max().mean())
        results[cat] = v
        print(f"{cat}: SEALED re-scorer = {v:.4f}   composite {BASE[cat]:.4f}   delta {v-BASE[cat]:+.4f}", flush=True)
    m = sum(results.values()) / 3
    bm = sum(BASE.values()) / 3
    print(f"\nMEAN: re-scorer {m:.4f}   composite {bm:.4f}   delta {m-bm:+.4f}   (leaderboard: GOA 0.325, FunBind 0.366, TransFew 0.381)", flush=True)


if __name__ == "__main__":
    main()
