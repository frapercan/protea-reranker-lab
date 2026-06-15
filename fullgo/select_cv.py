"""SELECT-internal held-out validation (NEVER touches the 227->230 TEST frame).

Splits the SELECT 220->227 eval proteins 50/50 by a stable hash, fits the per-category GBM on
split A, evaluates f_micro_w on split B (SELECT GT restricted to B). Compares feature sets:
  none -> KNN + 7-seed clf + self-prior            (the 0.381-class stack)
  v1   -> + association (TOPN80, rank a_all)        (the 0.390 champion's assoc)
  v2   -> + association (TOPN200, rank lift, +lift)  (the PK-tradeoff variant)
This decides which lever generalises WITHIN select, independent of any TEST seal.

Usage: select_cv.py none|v1|v2
"""
import sys, math, tempfile, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
from cafaeval.evaluation import cafa_eval

MODE = sys.argv[1] if len(sys.argv) > 1 else "none"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
SEL_KNN = "/tmp/select_knn_composite.tsv"; SEL_CLF = "/tmp/sel_m2_seedavg7.tsv"
SEL_GT = "/tmp/select_gt_{cat}.tsv"; SEL_POOL = "/tmp/v220_exp_aspect.tsv"
SEL_SP = "/tmp/select_selfprior_leaf.tsv"
SEL_AS = {"v1": "/tmp/assoc_sel.tsv", "v2": "/tmp/assoc_v2_sel.tsv"}.get(MODE)


def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return 0.0


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


def load_sp(path):
    by = defaultdict(dict)
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2 and c[1].startswith("GO:"): by[c[0]][c[1]] = 1.0
    return by


def load_assoc(path):
    by = defaultdict(dict)
    if not path: return by
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 5 and c[1].startswith("GO:"): by[c[0]][c[1]] = (fnum(c[2]), fnum(c[3]), fnum(c[4]))
        elif len(c) >= 4 and c[1].startswith("GO:"): by[c[0]][c[1]] = (fnum(c[2]), fnum(c[3]), 0.0)
    return by


def freq_pool(path):
    fr = defaultdict(int); seen = set()
    for line in open(path):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2:
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


def split(p):
    h = int(hashlib.md5(p.encode()).hexdigest(), 16)
    return "A" if h % 2 == 0 else "B"


def feats(knn, clf, sp, asc, p, t, use_assoc):
    kf = knn.get(p, {}).get(t, [0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    cs = clf.get(p, {}).get(t, 0.0); ss = sp.get(p, {}).get(t, 0.0)
    f = list(kf) + [cs, 1.0 if t in knn.get(p, {}) else 0.0, 1.0 if t in clf.get(p, {}) else 0.0,
                    ss, 1.0 if t in sp.get(p, {}) else 0.0]
    if use_assoc:
        aa, ax, al = asc.get(p, {}).get(t, (0.0, 0.0, 0.0))
        f += [aa, ax, al, 1.0 if t in asc.get(p, {}) else 0.0]
    return f


def main():
    par = parents_map(); cache = {}
    ia = {}
    for line in open(IA):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2: ia[c[0]] = fnum(c[1])
    knn, clf, sp = load_knn(SEL_KNN), load_clf(SEL_CLF), load_sp(SEL_SP)
    asc = load_assoc(SEL_AS); use_assoc = MODE in ("v1", "v2")
    freq = freq_pool(SEL_POOL)
    res = {}
    for cat in ("NK", "LK", "PK"):
        gt = gtset(SEL_GT.format(cat=cat), par, cache)
        trX, trY, teRows = [], [], []
        for p in gt:
            kn = knn.get(p, {}); cl = clf.get(p, {}); s = sp.get(p, {}); a = asc.get(p, {}) if use_assoc else {}
            cand = set(kn) | set(cl) | set(s) | set(a)
            sp_ = split(p)
            for t in cand:
                f = feats(knn, clf, sp, asc, p, t, use_assoc) + [ia.get(t, 0.0), math.log1p(freq.get(t, 0))]
                y = 1 if t in gt[p] else 0
                if sp_ == "A": trX.append(f); trY.append(y)
                else: teRows.append((p, t, f))
        b = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31,
                           min_data_in_leaf=50, feature_fraction=0.9, verbose=-1),
                      lgb.Dataset(np.array(trX, np.float32), label=np.array(trY, np.int32)),
                      num_boost_round=300)
        pred = b.predict(np.array([r[2] for r in teRows], np.float32))
        Bset = set(p for p in gt if split(p) == "B")
        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td) / "pred"; pdir.mkdir()
            with open(pdir / "m.tsv", "w") as w:
                for (p, t, _), s in zip(teRows, pred): w.write(f"{p}\t{t}\t{s:.6f}\n")
            gtf = Path(td) / "gt.tsv"
            with open(gtf, "w") as w, open(SEL_GT.format(cat=cat)) as fh:
                for line in fh:
                    c = line.rstrip("\n").split("\t")
                    if len(c) >= 2 and c[0] in Bset: w.write(line if line.endswith("\n") else line + "\n")
            df, _ = cafa_eval(OBO, str(pdir), str(gtf), ia=IA, prop="fill", norm="cafa",
                              no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
            sub = df.reset_index()
            col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
            res[cat] = float(sub.groupby("ns")[col].max().mean())
        print(f"  {cat}: {res[cat]:.4f}", flush=True)
    print(f"MODE={MODE}  SELECT-internal held-out MEAN = {sum(res.values())/3:.4f}", flush=True)


if __name__ == "__main__":
    main()
