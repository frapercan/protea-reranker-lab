"""CG.3 leakage ablation: does the clean-channel signal survive removing target-as-partner edges?

If a large share of added-true IA-mass came from partners that are THEMSELVES test targets, the
channel could be laundering target-to-target information. This recomputes clean-arm top5/top10
precision with ALL partners whose UniProt accession is in the combined LK+PK target set REMOVED.
Signal that holds under this exclusion is carried by non-target, independently-annotated partners.
Partner annotations are the frozen v227=t0 corpus in every case (temporally pre the v230 gains).
"""
import collections, gzip, time, json
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"; GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
STR = ROOT / "storage/string_v12"; ACC_TAX = ROOT / "storage/regen_headline/cg3_acc_taxon.tsv"
EDGE_CUT = 0.40
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stack.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iamass(ts): return float(sum(IA.get(t, 0.0) for t in ts))

meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id)); id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp); ref = ref.dropna(subset=["go"])
acc2bpraw = ref[ref.asp == "P"].groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
_bpprop = {}
def bp_prop(acc):
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc)
        r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset(); _bpprop[acc] = r
    return r
acc_tax = pd.read_csv(ACC_TAX, sep="\t", header=None, names=["accession", "taxon"], dtype=str)
tax_of = dict(zip(acc_tax.accession, acc_tax.taxon))
DL_TAXA = sorted({p.name.split(".")[0][3:] for p in STR.glob("tax*.protein.links.detailed.v12.0.txt.gz")})

# combined target set (both cells) -> the partners to drop in the ablation
allt = set()
for fn in ["groundtruth_LK.tsv", "groundtruth_PK.tsv"]:
    allt |= set(pd.read_csv(GTDIR / fn, sep="\t").query("aspect=='P'").EntryID.unique())
log(f"combined target set: {len(allt):,}")

def load_aliases(taxon):
    fn = STR / f"tax{taxon}.protein.aliases.v12.0.txt.gz"
    s2u = collections.defaultdict(set); u2s = collections.defaultdict(set)
    with gzip.open(fn, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3 or p[2] != "UniProt_AC": continue
            s2u[p[0]].add(p[1]); u2s[p[1]].add(p[0])
    return s2u, u2s

def collect_edges(taxon, target_sids):
    fn = STR / f"tax{taxon}.protein.links.detailed.v12.0.txt.gz"
    out = collections.defaultdict(list)
    with gzip.open(fn, "rt") as fh:
        fh.readline()
        for ln in fh:
            i = ln.find(" ")
            if ln[:i] not in target_sids: continue
            c = ln.split(" "); cx = int(c[5]); ex = int(c[6])
            if ex == 0 and cx == 0: continue
            out[ln[:i]].append((c[1], cx, ex))
    return out

def run(drop_target_partners):
    res = {}
    for cell, gt_fn in [("LK-BPO", "groundtruth_LK.tsv"), ("PK-BPO", "groundtruth_PK.tsv")]:
        gt = pd.read_csv(GTDIR / gt_fn, sep="\t").query("aspect=='P'").copy()
        gt["term"] = gt.term.map(lambda g: alt.get(g, g))
        gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
        targets = sorted(gt_leaf); target_set = set(targets)
        tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
        tb = tb[tb.protein_accession.isin(target_set)]; tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
        pool = tb[tb.go.isin(BP)].groupby("protein_accession").go.apply(set).to_dict()
        gt_prop = {p: frozenset().union(*[anc(t) for t in ts]) if ts else frozenset() for p, ts in gt_leaf.items()}
        pool_prop = {p: (frozenset().union(*[anc(t) for t in pool[p]]) if pool.get(p) else frozenset()) for p in targets}
        tax_targets = collections.defaultdict(list)
        for p in targets: tax_targets[tax_of.get(p)].append(p)
        score = {}
        for taxon in DL_TAXA:
            tps = [p for p in tax_targets.get(taxon, []) if p in target_set]
            if not tps: continue
            s2u, u2s = load_aliases(taxon)
            sid_of = {p: u2s.get(p, set()) for p in tps}
            tsids = set().union(*sid_of.values()) if sid_of else set()
            if not tsids: continue
            edges = collect_edges(taxon, tsids)
            sid2tacc = collections.defaultdict(set)
            for p in tps:
                for sid in sid_of[p]: sid2tacc[sid].add(p)
            for sid, elist in edges.items():
                for tacc in sid2tacc.get(sid, ()):
                    own = sid_of[tacc]; sc = score.setdefault(tacc, collections.defaultdict(float))
                    for (b, cx, ex) in elist:
                        if b in own: continue
                        w = 1.0 - (1.0 - ex / 1000.0) * (1.0 - cx / 1000.0)
                        if w < EDGE_CUT: continue
                        Bb = frozenset()
                        for pacc in s2u.get(b, ()):
                            if pacc == tacc: continue
                            if drop_target_partners and pacc in allt: continue   # <-- ablation
                            Bb = Bb | bp_prop(pacc)
                        for t in Bb: sc[t] += w
        # top5/top10 precision
        out = {}
        for K in (5, 10):
            at = af = 0.0
            for p in targets:
                sd = score.get(p)
                if not sd: continue
                present = pool.get(p, set()); gtp = gt_prop[p]; covered = set(pool_prop[p]); picks = 0
                for g, sv in sorted(sd.items(), key=lambda kv: -kv[1]):
                    if sv <= 0: break
                    if g in present: continue
                    picks += 1
                    new = anc(g) - covered
                    at += iamass(new & gtp); af += iamass(new - gtp); covered |= new
                    if picks >= K: break
            out[f"top{K}_prec"] = round(at / (at + af), 4) if (at + af) > 0 else None
            out[f"top{K}_true"] = round(at, 2)
        res[cell] = out
        log(f"  {cell} drop_target_partners={drop_target_partners}: {out}")
    return res

log("=== BASELINE (all partners) ===")
base = run(False)
log("=== ABLATION (target-as-partner removed) ===")
abl = run(True)
json.dump({"baseline_all_partners": base, "ablation_target_partners_removed": abl},
          open(ROOT / "storage/regen_headline/cg3_leakage_ablation.json", "w"), indent=2)
log("wrote cg3_leakage_ablation.json")
