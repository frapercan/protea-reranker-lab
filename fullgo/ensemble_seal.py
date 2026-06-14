"""Learned ensemble: GBM over KNN sub-features + classifier score + IA + freq.
Candidates = union(KNN terms, classifier terms) -> removes the KNN recall ceiling.
Train per-category on SELECT 220->227, SEAL on 7401 (exact harness).
Baselines: KNN 0.324, classifier 0.326, re-scorer 0.330.
"""
import math, tempfile
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
SEL_KNN = "/tmp/select_knn_composite.tsv"; SEL_CLF = "/tmp/sel_clf_pred.tsv"
SEL_GT = "/tmp/select_gt_{cat}.tsv"; SEL_POOL = "/tmp/v220_exp_aspect.tsv"
TEST_KNN = "/tmp/canon_composite.tsv"; TEST_CLF = "/tmp/m0_asl_pred.tsv"
TEST_GT = REL + "/groundtruth_{cat}.tsv"; TEST_FREQ = "/tmp/v227_exp_freq.tsv"
BASE = {"NK": 0.412, "LK": 0.394, "PK": 0.165}


def parents_map():
    par = defaultdict(set); cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]": cur = None
        elif line.startswith("id: GO:"): cur = line[4:]
        elif line.startswith("is_a:") and cur: par[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            p = line.split()
            if len(p) >= 3: par[cur].add(p[2])
    return par


def anc(t, par, c):
    if t in c: return c[t]
    o = set(); st = list(par.get(t, ()))
    while st:
        a = st.pop()
        if a in o: continue
        o.add(a); st.extend(par.get(a, ()))
    c[t] = o; return o


def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return 0.0


def load_knn(path):
    f = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 11 or not c[1].startswith("GO:"): continue
            k = (c[0], c[1]); sc = fnum(c[2])
            if k not in f or sc > f[k][0]:
                f[k] = [sc, fnum(c[3]), fnum(c[7]), fnum(c[8]), fnum(c[9]), fnum(c[10])]
    by = defaultdict(dict)
    for (p, t), ft in f.items(): by[p][t] = ft
    return by


def load_clf(path):
    by = defaultdict(dict)
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 3 and c[1].startswith("GO:"):
            s = fnum(c[2])
            if s > by[c[0]].get(c[1], 0.0): by[c[0]][c[1]] = s
    return by


def freq_pool(path, two=False):
    fr = defaultdict(int); seen = set()
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if two:
            if len(c) >= 2: fr[c[0]] = int(c[1])
        elif len(c) >= 2:
            k = (c[0], c[1])
            if k not in seen: seen.add(k); fr[c[1]] += 1
    return fr


def gtset(path, par, c):
    gt = defaultdict(set)
    for line in open(path):
        x = line.rstrip("\n").split("\t")
        if not x or x[0] == "EntryID" or len(x) < 2 or not x[1].startswith("GO:"): continue
        gt[x[0]].add(x[1]); gt[x[0]] |= anc(x[1], par, c)
    return gt


def rows_for(knn, clf, gt, ia, freq):
    rows = []
    for p in gt:
        gtp = gt[p]
        kn = knn.get(p, {}); cl = clf.get(p, {})
        for t in set(kn) | set(cl):
            kf = kn.get(t, [0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
            cs = cl.get(t, 0.0)
            feat = kf + [cs, 1.0 if t in kn else 0.0, 1.0 if t in cl else 0.0,
                         ia.get(t, 0.0), math.log1p(freq.get(t, 0))]
            rows.append((p, t, feat, 1 if t in gtp else 0))
    return rows


def main():
    par = parents_map(); cache = {}
    ia = {}
    for line in open(IA):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2: ia[c[0]] = fnum(c[1])
    sk, sc_ = load_knn(SEL_KNN), load_clf(SEL_CLF)
    tk, tc = load_knn(TEST_KNN), load_clf(TEST_CLF)
    sfreq, tfreq = freq_pool(SEL_POOL), freq_pool(TEST_FREQ, two=True)
    res = {}
    for cat in ("NK", "LK", "PK"):
        sgt = gtset(SEL_GT.format(cat=cat), par, cache)
        tgt = gtset(TEST_GT.format(cat=cat), par, cache)
        tr = rows_for(sk, sc_, sgt, ia, sfreq)
        te = rows_for(tk, tc, tgt, ia, tfreq)
        Xtr = np.array([r[2] for r in tr], np.float32); ytr = np.array([r[3] for r in tr], np.int32)
        Xte = np.array([r[2] for r in te], np.float32)
        b = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31,
                           min_data_in_leaf=50, feature_fraction=0.9, verbose=-1),
                      lgb.Dataset(Xtr, label=ytr), num_boost_round=300)
        pred = b.predict(Xte)
        with tempfile.TemporaryDirectory() as td:
            pd_ = Path(td) / "pred"; pd_.mkdir()
            with open(pd_ / "m.tsv", "w") as w:
                for (p, t, _, _), s in zip(te, pred): w.write(f"{p}\t{t}\t{s:.6f}\n")
            kw = dict(ia=IA, prop="fill", norm="cafa", no_orphans=True, max_terms=None,
                      th_step=0.01, n_cpu=1, toi_file=TOI)
            if cat == "PK": kw["exclude"] = PKK_TEST
            df, _ = cafa_eval(OBO, str(pd_), TEST_GT.format(cat=cat), **kw)
            sub = df.reset_index()
            col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
            v = float(sub.groupby("ns")[col].max().mean())
        res[cat] = v
        names = ["knn", "dist", "id_nw", "id_sw", "tax", "vote", "clf", "knn_p", "clf_p", "IA", "lfreq"]
        imp = {n: int(x) for n, x in zip(names, b.feature_importance("gain"))}
        print(f"{cat}: ENSEMBLE sealed = {v:.4f}   composite {BASE[cat]:.4f}   delta {v-BASE[cat]:+.4f}  imp={imp}", flush=True)
    m = sum(res.values()) / 3
    print(f"\nMEAN: ensemble {m:.4f}   (KNN 0.324 / classifier 0.326 / re-scorer 0.330 / FunBind 0.366 / TransFew 0.381)", flush=True)


if __name__ == "__main__":
    main()
