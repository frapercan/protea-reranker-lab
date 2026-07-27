"""Network (CG.3, STRING v12.0) proposal regenerator for the recall-ceiling union.

Emits, per LK/PK-BPO target, the propagated BP toi-index set proposed by its STRING partners'
FROZEN t0 (v227) BP annotations (clean arm = experimental + coexpression channels, combined
>= 0.40). Self / same-accession partners excluded. This is the MAXIMAL proposal set (every
partner-annotated BP term, no top-k cut) -- a recall ceiling, not a precision arm.

Leakage: STRING v12.0 release 2023-07-26 << t0 (2025-09-04); textmining channel NEVER read;
partner annotations only from frozen v227=t0 reference_annotations. Verbatim CG.3 machinery.
Output: recall_scratch/net_prop_clean.pkl = {accession: np.int32 toi-idx array}.
"""
import json, collections, time, gzip, pickle
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
F = ROOT / "storage/protea-frozen-v227-2025-09-04"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
STR = ROOT / "storage/string_v12"
ACC_TAX = ROOT / "storage/regen_headline/cg3_acc_taxon.tsv"
W = ROOT / "storage/regen_headline"
SC = W / "recall_scratch"; SC.mkdir(exist_ok=True)
TOI_FILE = str(ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt/groundtruth_terms_of_interest.txt")
EDGE_CUT = 0.40
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# obo
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, st = set(), [t]
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): st.append(p)
    r = frozenset(x for x in seen if x in BP); _anc[t] = r; return r

toi_go = sorted({g.strip() for g in open(TOI_FILE) if g.strip() in BP})
toi_idx = {g: i for i, g in enumerate(toi_go)}
def to_toi(terms):
    s = set()
    for t in terms:
        j = toi_idx.get(t)
        if j is not None: s.add(j)
    return s

# frozen t0 BP annotations per accession (v227)
meta = pq.read_table(F / "go_term_metadata.parquet").to_pandas()
id2go = dict(zip(meta.go_term_id, meta.go_id)); id2asp = dict(zip(meta.go_term_id, meta.aspect))
ref = pq.read_table(F / "reference_annotations.parquet", columns=["accession", "go_term_id"]).to_pandas()
ref["go"] = ref.go_term_id.map(id2go).map(lambda g: alt.get(g, g) if isinstance(g, str) else g)
ref["asp"] = ref.go_term_id.map(id2asp)
ref = ref.dropna(subset=["go"]); refP = ref[ref.asp == "P"]
acc2bpraw = refP.groupby("accession").go.apply(lambda s: frozenset(s)).to_dict()
_bpprop = {}
def bp_prop(acc):
    r = _bpprop.get(acc)
    if r is None:
        raw = acc2bpraw.get(acc)
        r = frozenset().union(*[anc(t) for t in raw]) if raw else frozenset()
        _bpprop[acc] = r
    return r
log(f"frozen BP: {len(acc2bpraw)} proteins")

# universe = atlas LK + PK
targets = set()
tcell = {}
for cell in ("lk", "pk"):
    for r in json.load(open(W / f"bp_atlas_perprotein_{cell}.json")):
        targets.add(r["protein"]); tcell[r["protein"]] = cell
tax_of = dict(zip(*[pd.read_csv(ACC_TAX, sep="\t", header=None, names=["a", "t"], dtype=str)[c] for c in ("a", "t")]))
DL_TAXA = sorted({p.name.split(".")[0][3:] for p in STR.glob("tax*.protein.links.detailed.v12.0.txt.gz")})
log(f"targets {len(targets)}; STRING taxa {len(DL_TAXA)}")

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
            i = ln.find(" "); a = ln[:i]
            if a not in target_sids: continue
            c = ln.split(" ")
            b = c[1]; cx = int(c[5]); ex = int(c[6]); db = int(c[7])
            if ex == 0 and cx == 0 and db == 0: continue
            out[a].append((b, cx, ex, db))
    return out

def noisy_or(*probs):
    r = 1.0
    for p in probs: r *= (1.0 - p)
    return 1.0 - r

score = collections.defaultdict(set)   # target acc -> set(GO terms) proposed (clean arm)
tax_targets = collections.defaultdict(list)
for p in targets: tax_targets[tax_of.get(p)].append(p)
for taxon in DL_TAXA:
    tps = [p for p in tax_targets.get(taxon, []) if p in targets]
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
        taccs = sid2tacc.get(sid)
        if not taccs: continue
        for tacc in taccs:
            own = sid_of[tacc]
            for (b, cx, ex, db) in elist:
                if b in own: continue
                Bb = frozenset()
                for pacc in s2u.get(b, ()):
                    if pacc == tacc: continue
                    Bb = Bb | bp_prop(pacc)
                if not Bb: continue
                if noisy_or(ex / 1000.0, cx / 1000.0) >= EDGE_CUT:
                    score[tacc] |= Bb
    log(f"taxon {taxon}: {len(tps)} targets; cumulative covered {len(score)}")

out = {}
for p, terms in score.items():
    s = to_toi(terms)
    if s: out[p] = np.fromiter(s, dtype=np.int32, count=len(s))
pickle.dump(out, open(SC / "net_prop_clean.pkl", "wb"))
log(f"wrote {len(out)} network proposal sets -> net_prop_clean.pkl")
