"""Build evidence-tiered training labels (arms) + per-protein t0 calibration features.

Inputs (frozen, v225): evidence_edges.tsv (acc, GO, evidence, aspect) parsed from goa_uniprot_all.gaf.225.gz
restricted to the generator_frames corpus accessions; the base SSE per-aspect meta (fixed vocab); the
generator_frames label matrix A (defines the 'full' positive set exactly); go-basic.obo (propagation).

Outputs under storage/sse_evidence/labels/:
  {asp}_rows.npy               corpus row indices into accs.json (SAME rows across arms == clean ablation)
  {asp}_full.npz / _noiea.npz / _exp.npz   sparse (rows x base-terms) label matrices; subsets of A by tier
Outputs under storage/sse_evidence/:
  calib_features.npz           per-acc v225 evidence summary (arm b calibration prior)
  build_report.json           reconciliation + coverage stats

Tiers: EXP experimental / PHY phylogenetic / COMP computational-sequence / AUTH author-curator / ELEC electronic(IEA).
noiea = A-positive supported by any NON-electronic tier (de-circularised). exp = A-positive supported by EXP tier.
"""
import json, time, collections
from pathlib import Path
import numpy as np, scipy.sparse as sp

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
ROOT = Path("/home/frapercan/Thesis2")
GF = ROOT / "storage/cooc_experiment/generator_frames"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
BASE = ROOT / "storage/sse_full"
GTDIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
OUT = ROOT / "storage/sse_evidence"; LAB = OUT / "labels"; LAB.mkdir(parents=True, exist_ok=True)
EDGES = OUT / "evidence_edges.tsv"

EXP = {"EXP","IDA","IPI","IMP","IGI","IEP","HTP","HDA","HMP","HGI","HEP"}
PHY = {"IBA","IBD","IKR","IRD"}
COMP = {"ISS","ISO","ISA","ISM","IGC","RCA"}
AUTH = {"TAS","NAS","IC"}
ELEC = {"IEA"}
BIT = {}
for s,b in [(EXP,1),(PHY,2),(COMP,4),(AUTH,8),(ELEC,16)]:
    for c in s: BIT[c] = b
NONELEC = 1|2|4|8
ASPECTS = {"mfo":"molecular_function","bpo":"biological_process","cco":"cellular_component"}
ASPLET = {"P":"bpo","F":"mfo","C":"cco"}

# ---- obo ----
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
    _anc[t] = seen; return seen

# ---- corpus ----
accs = json.load(open(GF / "accs.json")); acc_idx = {a: i for i, a in enumerate(accs)}
vocab = json.load(open(GF / "vocab.json"))
L = np.load(GF / "labels.npz")
A = sp.csr_matrix((L["data"], L["indices"], L["indptr"]), shape=tuple(L["shape"])).tocsr()
import pandas as pd
eval_prots = set()
for fn in ["groundtruth_LK.tsv", "groundtruth_PK.tsv"]:
    eval_prots |= set(pd.read_csv(GTDIR / fn, sep="\t").EntryID.unique())
log(f"corpus accs {len(accs)}  eval held-out {len(eval_prots)}")

# ---- load evidence edges: acc_row -> {term: tierbits} (direct annotations, v225) ----
direct = collections.defaultdict(dict)     # acc_row -> {go(canon): bits}
calib = collections.defaultdict(lambda: np.zeros(5, dtype=np.int64))   # acc_row -> [EXP,PHY,COMP,AUTH,ELEC] counts
calib_terms = collections.defaultdict(set)
term_tier = collections.defaultdict(lambda: np.zeros(5, dtype=np.int64))  # go(canon) -> direct tier counts (corpus)
TP = {1:0,2:1,4:2,8:3,16:4}
n_edge = 0; n_skip_acc = 0
with open(EDGES) as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        acc, go, ev, asp = p[0], p[1], p[2], p[3]
        r = acc_idx.get(acc)
        if r is None: n_skip_acc += 1; continue
        b = BIT.get(ev, 0)
        if b == 0: continue                         # ND / unknown evidence -> ignore
        go = alt.get(go, go)
        direct[r][go] = direct[r].get(go, 0) | b
        tierpos = TP[b]
        calib[r][tierpos] += 1
        calib_terms[r].add(go)
        term_tier[go][tierpos] += 1
        n_edge += 1
log(f"edges kept {n_edge}  skipped(acc not in corpus) {n_skip_acc}  proteins-with-evidence {len(direct)}")

# ---- calibration features (arm b): per acc, from ALL aspects' direct annotations (v225 KNOWN) ----
c_accrow = np.array(sorted(calib.keys()))
c_counts = np.stack([calib[r] for r in c_accrow])            # (n, 5) [EXP,PHY,COMP,AUTH,ELEC]
c_total = c_counts.sum(1)
c_expfrac = c_counts[:,0] / np.maximum(c_total, 1)
c_nonelecfrac = c_counts[:,:4].sum(1) / np.maximum(c_total, 1)
c_breadth = np.array([len(calib_terms[r]) for r in c_accrow])
np.savez(OUT / "calib_features.npz", acc_row=c_accrow,
         counts=c_counts, total=c_total, exp_frac=c_expfrac,
         nonelec_frac=c_nonelecfrac, breadth=c_breadth,
         has_exp=(c_counts[:,0] > 0).astype(np.int8),
         cols=np.array(["EXP","PHY","COMP","AUTH","ELEC"]))
log(f"calib features saved for {len(c_accrow)} proteins")

# ---- global per-term DIRECT evidence profile (corpus, v225) for arm-c + wall diagnostic ----
tt_terms = np.array(sorted(term_tier.keys()))
tt_counts = np.stack([term_tier[t] for t in tt_terms]) if len(tt_terms) else np.zeros((0,5), np.int64)
np.savez(OUT / "term_evidence_direct.npz", terms=tt_terms, counts=tt_counts,
         cols=np.array(["EXP","PHY","COMP","AUTH","ELEC"]))
log(f"term evidence profile saved for {len(tt_terms)} terms")

report = {"n_edges_kept": n_edge, "n_proteins_with_evidence": len(direct),
          "tier_bits": {"EXP":1,"PHY":2,"COMP":4,"AUTH":8,"ELEC":16}, "aspects": {}}

# ---- per aspect: fixed vocab = base meta terms; rows = base eligibility; build arm subsets of A ----
for asp, nsname in ASPECTS.items():
    meta = np.load(BASE / f"{asp}_meta.npz", allow_pickle=True)
    terms = [alt.get(g, g) for g in meta["terms"].tolist()]
    term_idx = {g: j for j, g in enumerate(terms)}; n_terms = len(terms)
    NSET = frozenset(t for t, n in ns.items() if n == nsname)
    ns_cols = np.array([i for i in range(len(vocab)) if vocab[i] in NSET])
    has = np.asarray(A[:, ns_cols].sum(1)).ravel() > 0
    elig = [i for i, a in enumerate(accs) if has[i] and a not in eval_prots]
    sub = np.array(sorted(elig))
    np.save(LAB / f"{asp}_rows.npy", sub)
    # A positives on this slice (== full arm, exact)
    # map vocab term -> base-term col
    vocab_canon = [alt.get(g, g) for g in vocab]
    col_of_vocab = np.array([term_idx.get(vocab_canon[i], -1) for i in range(len(vocab))])
    Asub = A[sub].tocoo()
    fr, fc, fd = [], [], []
    seen_full = set()
    for rr, cc in zip(Asub.row, Asub.col):
        bc = col_of_vocab[cc]
        if bc < 0: continue
        key = (rr, bc)
        if key in seen_full: continue
        seen_full.add(key); fr.append(rr); fc.append(bc)
    Yfull = sp.csr_matrix((np.ones(len(fr), np.int8), (fr, fc)), shape=(len(sub), n_terms))
    Yfull.data[:] = 1
    # tier support per (row, base-term) via propagation of the row's direct annotations
    supp = {}   # (row, col) -> bits
    for local, r in enumerate(sub):
        for go, bits in direct.get(r, {}).items():
            for a in anc(go):
                bc = term_idx.get(a)
                if bc is not None:
                    supp[(local, bc)] = supp.get((local, bc), 0) | bits
    # arm subsets: keep A-positive iff support has the required tier
    def subset(mask_fn):
        rr, cc = [], []
        Yf = Yfull.tocoo()
        for r, c in zip(Yf.row, Yf.col):
            b = supp.get((r, c), 0)
            if mask_fn(b): rr.append(r); cc.append(c)
        return sp.csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(len(sub), n_terms))
    Ynoiea = subset(lambda b: (b & NONELEC) != 0)
    Yexp = subset(lambda b: (b & 1) != 0)
    # reconciliation: how many A-positives have ANY GAF tier support
    Yf = Yfull.tocoo(); nA = Yf.nnz
    n_supp = sum(1 for r, c in zip(Yf.row, Yf.col) if supp.get((r, c), 0) != 0)
    for nm, Y in [("full", Yfull), ("noiea", Ynoiea), ("exp", Yexp)]:
        sp.save_npz(LAB / f"{asp}_{nm}.npz", Y.tocsr())
    report["aspects"][asp] = {"n_rows": int(len(sub)), "n_terms": n_terms,
        "A_positives": int(nA), "A_pos_with_gaf_support": int(n_supp),
        "gaf_support_frac": round(n_supp / max(nA, 1), 4),
        "noiea_positives": int(Ynoiea.nnz), "exp_positives": int(Yexp.nnz),
        "noiea_frac_of_full": round(Ynoiea.nnz / max(nA, 1), 4),
        "exp_frac_of_full": round(Yexp.nnz / max(nA, 1), 4)}
    log(f"{asp}: rows {len(sub)} terms {n_terms} | A+ {nA} gaf-supp {n_supp/max(nA,1):.3f} "
        f"| noiea {Ynoiea.nnz} ({Ynoiea.nnz/max(nA,1):.3f}) exp {Yexp.nnz} ({Yexp.nnz/max(nA,1):.3f})")

json.dump(report, open(OUT / "build_report.json", "w"), indent=1)
log("BUILD LABELS DONE")
print(json.dumps(report, indent=1))
