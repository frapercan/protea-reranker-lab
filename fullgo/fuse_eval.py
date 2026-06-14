"""Learned KNN + self-prior fusion, scored with cafaeval (exact harness).

Usage:
  fuse_eval.py --knn knn.tsv --selfprior sp.tsv --gtdir DIR --ia IA --obo OBO \
     --alpha A [--toi TOI] [--pkknown PKK] [--cats NK,LK,PK]

knn.tsv: protein<TAB>go<TAB>score (+header, extra cols ok)
sp.tsv:  protein<TAB>go            (self-prior leaf, implicit score 1.0)
gtdir:   contains groundtruth_{NK,LK,PK}.tsv  (or select_gt_{cat}.tsv via --gtpat)
Fusion (maxblend): blended = max(knn_score, alpha * selfprior)  -- alpha in [0,1]
"""
import argparse
import tempfile
from collections import defaultdict
from pathlib import Path


def read_knn(path):
    by = defaultdict(dict)
    with open(path) as fh:
        first = fh.readline().split("\t")
        # detect header
        def emit(c):
            if len(c) >= 3 and c[1].startswith("GO:"):
                try:
                    s = float(c[2])
                except ValueError:
                    return
                p, g = c[0], c[1]
                if s > by[p].get(g, 0.0):
                    by[p][g] = s
        if first and first[1:2] and first[1].startswith("GO:"):
            emit([x.strip() for x in first])
        for line in fh:
            emit([x.strip() for x in line.split("\t")])
    return by


def read_sp(path, scored=False):
    if scored:
        by = defaultdict(dict)
        with open(path) as fh:
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 3 and c[1].startswith("GO:"):
                    try:
                        s = float(c[2])
                    except ValueError:
                        continue
                    if s > by[c[0]].get(c[1], 0.0):
                        by[c[0]][c[1]] = s
        return by
    by = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2 and c[1].startswith("GO:"):
                by[c[0]].add(c[1])
    return by


def gt_prots(gtfile):
    prots = set()
    with open(gtfile) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if c and c[0] == "EntryID":
                continue
            if len(c) >= 2 and c[1].startswith("GO:"):
                prots.add(c[0])
    return prots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", required=True)
    ap.add_argument("--selfprior", required=True)
    ap.add_argument("--gtdir", required=True)
    ap.add_argument("--gtpat", default="groundtruth_{cat}.tsv")
    ap.add_argument("--ia", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--toi", default=None)
    ap.add_argument("--pkknown", default=None)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--mode", default="maxblend", choices=["maxblend", "agreement", "sponly"])
    ap.add_argument("--sp-scored", action="store_true", help="second stream carries graded scores in col3")
    ap.add_argument("--cats", default="NK,LK,PK")
    ap.add_argument("--label", default="fuse")
    args = ap.parse_args()

    from cafaeval.evaluation import cafa_eval

    knn = read_knn(args.knn)
    sp = read_sp(args.selfprior, scored=args.sp_scored)
    gtdir = Path(args.gtdir)
    a = args.alpha

    vals = {}
    for cat in args.cats.split(","):
        gtfile = gtdir / args.gtpat.format(cat=cat)
        prots = gt_prots(gtfile)
        with tempfile.TemporaryDirectory() as td:
            pd_ = Path(td) / "pred"
            pd_.mkdir()
            out = pd_ / f"{args.label}.tsv"
            n = 0
            with open(out, "w") as w:
                for p in prots:
                    cand = {}
                    spp = sp.get(p, {})
                    def sp_iter(spp):
                        if isinstance(spp, dict):
                            return spp.items()
                        return ((g, 1.0) for g in spp)
                    if args.mode == "sponly":
                        for g, s in sp_iter(spp):
                            cand[g] = s
                    else:
                        for g, s in knn.get(p, {}).items():
                            cand[g] = s
                        if args.mode == "agreement":
                            if a > 0:
                                spkeys = set(spp.keys()) if isinstance(spp, dict) else spp
                                for g in list(cand.keys()):
                                    if g in spkeys:
                                        cand[g] = min(1.0, cand[g] + a)
                        else:  # maxblend
                            if a > 0:
                                for g, sps in sp_iter(spp):
                                    v = a * sps
                                    if v > cand.get(g, 0.0):
                                        cand[g] = v
                    for g, s in cand.items():
                        w.write(f"{p}\t{g}\t{s:.6f}\n")
                        n += 1
            if n == 0:
                vals[cat] = None
                continue
            kw = dict(ia=args.ia, prop="fill", norm="cafa", no_orphans=True,
                      max_terms=None, th_step=0.01, n_cpu=1)
            if args.toi:
                kw["toi_file"] = args.toi
            if cat == "PK" and args.pkknown:
                kw["exclude"] = args.pkknown
            df, _ = cafa_eval(args.obo, str(pd_), str(gtfile), **kw)
            sub = df.reset_index()
            col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
            vals[cat] = float(sub.groupby("ns")[col].max().mean())
    parts = " ".join(f"{c}={vals[c]:.4f}" if vals[c] is not None else f"{c}=NA" for c in vals)
    m = [v for v in vals.values() if v is not None]
    print(f"alpha={a:.2f}  {parts}  MEAN={sum(m)/len(m):.4f}")


if __name__ == "__main__":
    main()
