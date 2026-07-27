"""CG.3 STEP 2 -- does unioning the clean-channel network proposals into the deployed pool
raise f_micro_w in the TRUE board frame?

ONE VARIABLE: candidate set only.
  Arm A  = deployed percut_rerank prediction file, VERBATIM (the pool the gate measured against).
  Arm B  = A + network BP extras (clean channels: experimental+coexpression, textmining/database
           NOT used), the top-k proposals per protein NOT already in the pool, scored by their own
           network confidence (min-max normalised per cell -> the generator's own score, the DB-free
           analog of the template's arm C "raw sigmoid"; no live DB, no transfer scorer).

TRUE frame: lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds -known (evaluation.nf:279).
cafa_eval run via repositories/PROTEA/.venv/bin/python in a detached driver.

Network generator recomputed here VERBATIM from cg3_step1_gate.py (clean arm only).
"""
import json, collections, time, gzip, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
PREDDIR = R / "percut_rerank/predictions"
GTDIR = R / "lafa_gt"
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
TOI = str(GTDIR / "groundtruth_terms_of_interest.txt")
STR = ROOT / "storage/string_v12"
ACC_TAX = ROOT / "storage/regen_headline/cg3_acc_taxon.tsv"
OUT = ROOT / "storage/regen_headline"
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")
EDGE_CUT = 0.40
EXTRA_KS = [5, 10, 25]


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


# obo (BP set + parents + alt) ----------------------------------------------------
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
log(f"obo: {len(BP):,} BP terms")

# frozen partner BP sets ----------------------------------------------------------
meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id)); id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
ref = ref.dropna(subset=["go"])
acc2bpraw = ref[ref.asp == "P"].groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
_bpprop = {}
def bp_prop(acc):
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc)
        r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset()
        _bpprop[acc] = r
    return r
log(f"partner BP sets: {len(acc2bpraw):,} proteins")

acc_tax = pd.read_csv(ACC_TAX, sep="\t", header=None, names=["accession", "taxon"], dtype=str)
tax_of = dict(zip(acc_tax.accession, acc_tax.taxon))
DL_TAXA = sorted({p.name.split(".")[0][3:] for p in STR.glob("tax*.protein.links.detailed.v12.0.txt.gz")})


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
            c = ln.split(" ")
            cx = int(c[5]); ex = int(c[6])          # coexpression, experimental (CLEAN channels)
            if ex == 0 and cx == 0: continue
            out[ln[:i]].append((c[1], cx, ex))
    return out


def clean_scores(targets):
    """term-score dict per target from CLEAN channels (exp+coexp), edge prob >= EDGE_CUT."""
    target_set = set(targets)
    tax_targets = collections.defaultdict(list)
    for p in targets: tax_targets[tax_of.get(p)].append(p)
    score = {}
    for taxon in DL_TAXA:
        tps = [p for p in tax_targets.get(taxon, []) if p in target_set]
        if not tps: continue
        s2u, u2s = load_aliases(taxon)
        sid_of = {p: u2s.get(p, set()) for p in tps}
        target_sids = set().union(*sid_of.values()) if sid_of else set()
        if not target_sids: continue
        edges = collect_edges(taxon, target_sids)
        sid2tacc = collections.defaultdict(set)
        for p in tps:
            for sid in sid_of[p]: sid2tacc[sid].add(p)
        for sid, elist in edges.items():
            for tacc in sid2tacc.get(sid, ()):
                own = sid_of[tacc]
                sc = score.setdefault(tacc, collections.defaultdict(float))
                for (b, cx, ex) in elist:
                    if b in own: continue
                    w = 1.0 - (1.0 - ex / 1000.0) * (1.0 - cx / 1000.0)
                    if w < EDGE_CUT: continue
                    Bb = frozenset()
                    for pacc in s2u.get(b, ()):
                        if pacc == tacc: continue
                        Bb = Bb | bp_prop(pacc)
                    for t in Bb: sc[t] += w
    return score


DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs = cafa_eval("{obo}", "{pd}", "{gt}", ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, toi_file="{toi}", exclude={known}, max_terms=None, th_step=0.01,
    n_cpu=6, weighted_only=False)
out = {{}}
for k, v in dfs.items(): out[k] = v.reset_index().to_dict(orient="records")
json.dump(out, open("{o}", "w"), default=str)
'''


def cafa(rows, gt_file, known):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows:
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        knownrepr = f'"{known}"' if known else "None"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=gt_file, ia=IA_F, toi=TOI,
                                     known=knownrepr, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print(r.stderr[-2000:], flush=True); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


results = {}
for cell, cat, gt_fn, known in [
    ("LK-BPO", "lk", str(GTDIR / "groundtruth_LK.tsv"), None),
    ("PK-BPO", "pk", str(GTDIR / "groundtruth_PK.tsv"), str(GTDIR / "groundtruth_PK_known.tsv")),
]:
    log(f"===== {cell} =====")
    # deployed pool rows (arm A) = the cell's prediction file VERBATIM
    dep = pd.read_csv(PREDDIR / cat / f"{cat}.tsv", sep="\t", header=None,
                      names=["prot", "term", "score"])
    dep["term"] = dep.term.map(lambda g: alt.get(g, g))
    base = list(dep.itertuples(index=False, name=None))
    # BP pool membership per protein (to exclude already-present terms from extras)
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    targets = sorted(pd.read_csv(gt_fn, sep="\t").query("aspect=='P'").EntryID.unique())
    tb = tb[tb.protein_accession.isin(set(targets))]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    pool_bp = tb[tb.go.isin(BP)].groupby("protein_accession").go.apply(set).to_dict()

    scores = clean_scores(targets)
    # min-max normalise network score per cell (across all proposed extra terms)
    allv = [v for p in scores for v in scores[p].values()]
    lo, hi = (min(allv), max(allv)) if allv else (0.0, 1.0)
    def norm(v): return (v - lo) / (hi - lo) if hi > lo else 0.5
    log(f"  targets {len(targets)}; deployed rows {len(base):,}; net-scored targets {len(scores):,}")

    fa = cafa(base, gt_fn, known)
    log(f"  A pool-only (TRUE frame{' -known' if known else ''}): {fa}")
    cell_res = {"A_pool_only_trueframe": fa, "deployed_reference": {"PK-BPO": 0.11666, "LK-BPO": 0.34829}[cell],
                "extras": {}}
    for K in EXTRA_KS:
        ex = []
        for p in targets:
            sd = scores.get(p)
            if not sd: continue
            present = pool_bp.get(p, set())
            picks = 0
            for g, sv in sorted(sd.items(), key=lambda kv: -kv[1]):
                if sv <= 0: break
                if g in present: continue
                ex.append((p, g, norm(sv)))
                picks += 1
                if picks >= K: break
        rowsB = base + ex
        fb = cafa(rowsB, gt_fn, known)
        d = round(fb - fa, 5) if (fb is not None and fa is not None) else None
        cell_res["extras"][f"top{K}"] = {"n_extras": len(ex), "B_pool_plus_extras": fb, "delta_B_minus_A": d}
        log(f"  top{K}: +{len(ex):,} extras -> B {fb}  (delta {d:+.5f})" if d is not None
            else f"  top{K}: B FAILED")
        json.dump(results | {cell: cell_res}, open(OUT / "cg3_step2_cafaeval.json", "w"), indent=2)
    results[cell] = cell_res
    json.dump(results, open(OUT / "cg3_step2_cafaeval.json", "w"), indent=2)

json.dump(results, open(OUT / "cg3_step2_cafaeval.json", "w"), indent=2)
log("wrote cg3_step2_cafaeval.json")
print(json.dumps(results, indent=2))
