"""Replicate the EXACT published CAFA_forever cafaeval invocation on our predictions.

Published flags (modules/local/evaluation.nf EVALUATE_{NK,LK,PK}):
  cafaeval <t0 go-basic.obo> <preds> <gt> -ia IA.tsv -toi TOI -prop fill -norm cafa -no_orphans [-known pk_known]
"""
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")

USE_TOI = "--toi" in sys.argv
PRED = sys.argv[sys.argv.index("--pred") + 1]
LABEL = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "m"


def load_gt(cat):
    prots = set()
    with open(REL / f"groundtruth_{cat}.tsv") as fh:
        next(fh, None)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2 and c[1].startswith("GO:"):
                prots.add(c[0])
    known = set()
    if cat == "PK":
        with open(REL / "groundtruth_PK_known.tsv") as fh:
            next(fh, None)
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 2:
                    known.add((c[0], c[1]))
    return prots, known


def read_pred(path):
    by = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 3 and c[1].startswith("GO:"):
                by[c[0]].append((c[1], c[2]))
    return by


def score(pred_by, cat):
    from cafaeval.evaluation import cafa_eval
    prots, known = load_gt(cat)
    with tempfile.TemporaryDirectory() as td:
        pd_ = Path(td) / "pred"
        pd_.mkdir()
        out = pd_ / f"{LABEL}.tsv"
        n = 0
        with open(out, "w") as w:
            for p in prots:
                for term, sc in pred_by.get(p, ()):
                    w.write(f"{p}\t{term}\t{sc}\n")
                    n += 1
        if n == 0:
            return None
        kw = dict(ia=IA, prop="fill", norm="cafa", no_orphans=True,
                  max_terms=None, th_step=0.01, n_cpu=1)
        if USE_TOI:
            kw["toi_file"] = TOI
        if cat == "PK":
            kw["exclude"] = str(REL / "groundtruth_PK_known.tsv")
        df, _ = cafa_eval(OBO, str(pd_), str(REL / f"groundtruth_{cat}.tsv"), **kw)
        sub = df.reset_index()
        col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
        return float(sub.groupby("ns")[col].max().mean())


pred_by = read_pred(PRED)
print(f"harness: toi={'YES' if USE_TOI else 'no'} no_orphans=True prop=fill norm=cafa max_terms=None th=0.01")
vals = {}
for cat in ("NK", "LK", "PK"):
    v = score(pred_by, cat)
    vals[cat] = v
    print(f"  {cat}: {v:.4f}")
print(f"  MEAN: {sum(vals.values())/3:.4f}")
