"""Stage 0 data prep for the contrastive seq->annotation aligner.
Builds and caches: BP ontology closure, IA, BP text-emb vocab, and per-train-protein
positive term index lists for arm A (EXP-only) and arm B (EXP+IEA). Read-only w.r.t. repos.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment/seq2ann_infonce")
FROZEN = Path("/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")

EXP = {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC","HTP","HDA","HMP","HGI","HEP"}

# ---- ontology ----
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
AC = {}
def anc(t):
    if t in AC: return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o: o.add(p); st.append(p)
    AC[t] = o; return o
def closure(terms):
    o = set()
    for g in terms:
        o.add(g); o |= anc(g)
    return o
IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
print(f"[{time.time()-t0:.0f}s] ontology BP {len(BP):,}, IA {len(IA):,}", flush=True)

# ---- BP text-emb vocab (annotation tower input) ----
gt = np.load(SC / "go_text_emb.npz", allow_pickle=True)
go_ids_all = np.array([alt.get(g, g) for g in gt["go_ids"].tolist()])
emb_all = gt["emb"].astype(np.float32)
# dedup after alt-mapping, keep BP only
seen = {}; keep_idx = []
for i, g in enumerate(go_ids_all):
    if g in BP and g not in seen:
        seen[g] = len(keep_idx); keep_idx.append(i)
keep_idx = np.array(keep_idx)
vocab = go_ids_all[keep_idx]
vocab_emb = emb_all[keep_idx]                          # (Nterm,768)
term2idx = {g: i for i, g in enumerate(vocab)}
vocab_ia = np.array([IA.get(g, 0.0) for g in vocab], np.float32)
print(f"[{time.time()-t0:.0f}s] BP text vocab {len(vocab):,} terms", flush=True)

# ---- reference annotations -> per-protein direct BP terms by evidence ----
meta = pq.read_table(FROZEN / "go_term_metadata.parquet")
id2go = {i: g for i, g in zip(meta.column("go_term_id").to_pylist(), meta.column("go_id").to_pylist())}
ra = pq.read_table(FROZEN / "reference_annotations.parquet")
acc = ra.column("accession").to_pylist()
gid = ra.column("go_term_id").to_pylist()
ev = ra.column("evidence_code").to_pylist()
exp_direct = collections.defaultdict(set)   # protein -> direct BP terms (EXP)
iea_direct = collections.defaultdict(set)   # protein -> direct BP terms (IEA)
for a, gi, e in zip(acc, gid, ev):
    g = id2go.get(gi)
    if not g: continue
    g = alt.get(g, g)
    if g not in BP: continue
    if e in EXP: exp_direct[a].add(g)
    elif e == "IEA": iea_direct[a].add(g)
print(f"[{time.time()-t0:.0f}s] EXP-direct prots {len(exp_direct):,}, IEA-direct prots {len(iea_direct):,}", flush=True)

# ---- restrict to train proteins with a code; build closure positives in vocab space ----
tr = np.load(SC / "clf_protein_codes.npz", allow_pickle=True)
tr_accs = tr["accs"]; tr_codes = tr["codes"]              # (88212,2048) float16
acc2row = {a: i for i, a in enumerate(tr_accs.tolist())}

def pos_indices(direct_map, protein):
    cl = closure(direct_map.get(protein, set())) & BP
    return sorted({term2idx[g] for g in cl if g in term2idx})

armA_pos = {}; armB_pos = {}
nA_pairs = nB_pairs = 0
for a in tr_accs.tolist():
    pa = pos_indices(exp_direct, a)
    pb = pos_indices({k: exp_direct[k] | iea_direct.get(k, set()) for k in [a]}, a)
    if pa: armA_pos[a] = pa; nA_pairs += len(pa)
    if pb: armB_pos[a] = pb; nB_pairs += len(pb)

# raw (pre-closure) direct pair counts over train, BP only
rawA = sum(len(exp_direct[a] & BP) for a in tr_accs.tolist())
rawB = sum(len((exp_direct[a] | iea_direct.get(a, set())) & BP) for a in tr_accs.tolist())
print(f"[{time.time()-t0:.0f}s] armA prots {len(armA_pos):,} closure-pairs {nA_pairs:,} (raw-direct {rawA:,})", flush=True)
print(f"[{time.time()-t0:.0f}s] armB prots {len(armB_pos):,} closure-pairs {nB_pairs:,} (raw-direct {rawB:,})", flush=True)

# ---- cache ----
np.savez(W / "vocab.npz", vocab=vocab, vocab_emb=vocab_emb, vocab_ia=vocab_ia)
# store positives as ragged via json of lists (indices)
json.dump({"armA": armA_pos, "armB": armB_pos}, open(W / "positives.json", "w"))
# store train codes rows we need (union of arm B prots) as float16
useprots = sorted(set(armB_pos) | set(armA_pos))
rows = np.array([acc2row[a] for a in useprots])
np.savez(W / "train_codes.npz", accs=np.array(useprots), codes=tr_codes[rows])
audit = {
    "EXP_evidence_set": sorted(EXP),
    "bp_text_vocab_terms": int(len(vocab)),
    "train_proteins_total": int(len(tr_accs)),
    "armA_EXP": {"proteins": len(armA_pos), "closure_pairs": nA_pairs, "raw_direct_pairs": rawA},
    "armB_EXP_IEA": {"proteins": len(armB_pos), "closure_pairs": nB_pairs, "raw_direct_pairs": rawB},
    "ia_ge4_vocab_terms": int((vocab_ia >= 4).sum()),
    "ia_ge6_vocab_terms": int((vocab_ia >= 6).sum()),
}
json.dump(audit, open(W / "prep_audit.json", "w"), indent=1)
print(json.dumps(audit, indent=1), flush=True)
print(f"[{time.time()-t0:.0f}s] DONE prep", flush=True)
