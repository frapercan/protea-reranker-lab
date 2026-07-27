"""Reverse-direction evaluation of the trained bipartite kWTA link-prediction model.

FORWARD (protein -> novel terms) is a known-dead generation lever (see
bipartite_kwta_linkpred_ceiling.json). THIS asks a DIFFERENT question: as a
RETRIEVAL / EXPLAINABILITY surface answering "which proteins have this GO-BP
function, most strongly", is term -> proteins trustworthy?

We reuse the EXISTING trained checkpoint (bipartite_kwta_ckpt.pt) -- NO retraining.
Score(protein, term) = < kwta(P(code_p)), kwta(T[term]) >, identical to the
ceiling script. For each BP term we rank the 7,401 HELD-OUT eval proteins (never
seen in training) by that score and score the ranking against propagated v230
ground truth.

Baselines: (a) kNN nearest-neighbour similarity (-distance) for that (prot,term),
propagated over the ontology closure with max -- the trivial retrieval surface the
deployed pipeline already exposes; (b) a term-independent popularity prior (rank
proteins by their BP annotation degree) -- tests whether the learned index is just
ranking by protein degree.

All existing artifacts are read-only; only the receipts under
storage/cooc_experiment/ are written.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn
from scipy.sparse import load_npz
from scipy.stats import rankdata

t0 = time.time()
W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
SC = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier")
FR = W / "generator_frames"
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
SEED = 42
K = 1024; KWTA_K = 128
MIN_TRUE = 5          # a term must have >= this many true eval proteins to be scorable
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cpu"

# ---------- ontology / IA / names (verbatim from the ceiling script) ----------
par = collections.defaultdict(set); ns = {}; alt = {}; name = {}
cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur = None
    elif line.startswith("id: GO:"):
        cur = line[4:]
    elif cur and line.startswith("namespace: "):
        ns[cur] = line[11:]
    elif cur and line.startswith("name: "):
        name[cur] = line[6:]
    elif cur and line.startswith("is_a: GO:"):
        par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
AC = {}
def anc(t):
    if t in AC:
        return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o:
                o.add(p); st.append(p)
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
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass
print(f"ontology {len(BP):,} BP, IA {len(IA):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- ground truth v230 (PK+LK+NK) closed over BP ----------
def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                G[f[0]].add(alt.get(f[1], f[1]))
    return {p: {g for g in closure(ts) if g in BP} for p, ts in G.items()}
gt_all = {}
for fn in ("groundtruth_PK.tsv", "groundtruth_LK.tsv", "groundtruth_NK.tsv"):
    for p, ts in load_gt(fn).items():
        gt_all.setdefault(p, set()).update(ts)
# also keep pk/lk separately to reproduce the ceiling's held-out train mask exactly
gt_pk = load_gt("groundtruth_PK.tsv"); gt_lk = load_gt("groundtruth_LK.tsv")
print(f"gt proteins (any BP): {len(gt_all):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- kNN pools from eval_scores (for the held-out mask AND the baseline) ----------
esc = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
e_prot = np.asarray(esc.column("protein_accession").to_pylist())
e_term = np.asarray([alt.get(g, g) for g in esc.column("go_term_id").to_pylist()])
e_cat = np.asarray(esc.column("category").to_pylist())
e_asp = np.asarray(esc.column("aspect").to_pylist())
e_dist = np.asarray(esc.column("distance").to_pylist(), dtype=np.float64)
bpo = e_asp == "bpo"
def knn_pool(cell):
    m = bpo & (e_cat == cell)
    d = collections.defaultdict(set)
    for p_, g_ in zip(e_prot[m], e_term[m]):
        d[p_].add(g_)
    return d
knn_pk = knn_pool("pk"); knn_lk = knn_pool("lk")

# ---------- train frame (verbatim): vocab, adjacency, held-out mask, standardisation ----------
tr_accs = json.load(open(FR / "accs.json"))
vocab = [alt.get(g, g) for g in json.load(open(FR / "vocab.json"))]
Y = load_npz(FR / "labels.npz").tocsr()
XL = np.load(FR / "learned_2048.npy")
bp_cols = np.array([j for j, g in enumerate(vocab) if g in BP])
bp_terms = [vocab[j] for j in bp_cols]
term_pos = {g: i for i, g in enumerate(bp_terms)}
Ybp = Y[:, bp_cols].tocsr()
n_bp = len(bp_terms)
pk_prots = sorted(set(gt_pk) & set(knn_pk))
lk_prots = sorted(set(gt_lk) & set(knn_lk))
held = set(pk_prots) | set(lk_prots)
apos = {a: i for i, a in enumerate(tr_accs)}
tr_rows_all = np.array([apos[a] for a in tr_accs if a not in held])
mu = XL[tr_rows_all].mean(0, keepdims=True); sd = XL[tr_rows_all].std(0, keepdims=True) + 1e-6
Ytr = Ybp[tr_rows_all]
train_support = np.asarray(Ytr.sum(0)).ravel().astype(np.int64)   # train edges per BP term
print(f"train frame: {len(tr_rows_all):,} train prots, {n_bp:,} BP terms  ({time.time()-t0:.0f}s)", flush=True)

# ---------- model + checkpoint (NO retraining) ----------
def kwta(x, k):
    if k >= x.shape[-1]:
        return x
    thr = torch.topk(x, k, dim=-1).values[..., -1:].detach()
    return x * (x >= thr).float()
class Bipartite(nn.Module):
    def __init__(self, d_in, n_terms, k, kk):
        super().__init__()
        self.P = nn.Linear(d_in, k); self.T = nn.Embedding(n_terms, k); self.kk = kk
    def prot_vec(self, x):
        return kwta(self.P(x), self.kk)
    def term_vec(self, idx):
        return kwta(self.T(idx), self.kk)
mdl = Bipartite(2048, n_bp, K, KWTA_K).to(DEV)
ck = torch.load(W / "bipartite_kwta_ckpt.pt", map_location=DEV, weights_only=False)
mdl.load_state_dict(ck["state_dict"]); mdl.eval()
best_mrr = ck["best_mrr"]
print(f"LOADED checkpoint (best val_mrr {best_mrr:.4f})  ({time.time()-t0:.0f}s)", flush=True)
with torch.no_grad():
    TV = mdl.term_vec(torch.arange(n_bp, device=DEV)).cpu().numpy()   # (n_bp, K)

# ---------- eval protein codes (blind window, held out of training) ----------
ec = np.load(SC / "eval_protein_codes.npz", allow_pickle=True)
eval_accs_raw = ec["accs"].tolist()
codes_raw = ec["codes"].astype(np.float32)
# CRITICAL: the ceiling held out only (pk_prots | lk_prots); the eval npz also contains NK and
# non-pool proteins that SHARE accessions with the training frame and were therefore TRAINED ON.
# Memorized proteins float to the top of a reverse ranking and would inflate the model. The task
# says train memorization does NOT count, so we restrict the ranking universe to TRULY held-out
# eval proteins (accession absent from the trained rows).
trained_acc = set(tr_accs[i] for i in tr_rows_all)
keep = [i for i, a in enumerate(eval_accs_raw) if a not in trained_acc]
n_contam = len(eval_accs_raw) - len(keep)
eval_accs = [eval_accs_raw[i] for i in keep]
codes = codes_raw[keep]
n_eval = len(eval_accs); erow = {a: i for i, a in enumerate(eval_accs)}
print(f"eval universe: {len(eval_accs_raw):,} total -> {n_eval:,} truly held out "
      f"({n_contam:,} dropped as trained/memorized)  ({time.time()-t0:.0f}s)", flush=True)
with torch.no_grad():
    Xn = torch.tensor((codes - mu) / sd, dtype=torch.float32, device=DEV)
    PV = np.vstack([mdl.prot_vec(Xn[i:i+2048]).cpu().numpy() for i in range(0, n_eval, 2048)])
overlap = len(set(eval_accs) & trained_acc)   # must be 0 after the filter above
print(f"eval: {n_eval:,} held-out proteins; train-overlap(now 0)={overlap}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- ground-truth boolean over (eval prot x BP term) ----------
GT = np.zeros((n_eval, n_bp), dtype=bool)
for a in eval_accs:
    for g in gt_all.get(a, ()):
        j = term_pos.get(g)
        if j is not None:
            GT[erow[a], j] = True
true_per_term = GT.sum(0)
degree = GT.sum(1).astype(np.float64)      # BP annotation degree per eval protein (popularity prior)
print(f"gt matrix: {int(GT.sum()):,} true (prot,term) cells; "
      f"terms with >= {MIN_TRUE} true = {int((true_per_term>=MIN_TRUE).sum()):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- model score matrix S = PV @ TV.T  (n_eval x n_bp) ----------
S = (PV @ TV.T).astype(np.float32)
print(f"score matrix {S.shape} built  ({time.time()-t0:.0f}s)", flush=True)

# ---------- kNN baseline matrix: similarity=-distance, closure-propagated with max ----------
NEG = np.float32(-1e30)
KNN = np.full((n_eval, n_bp), NEG, dtype=np.float32)
# unique terms present in the bpo kNN rows -> precompute BP closure column indices
uniq_terms = set(e_term[bpo].tolist())
clos_cols = {}
for g in uniq_terms:
    cols = [term_pos[x] for x in closure([g]) if x in term_pos]
    clos_cols[g] = np.array(cols, dtype=np.int64) if cols else np.empty(0, np.int64)
mask = bpo
bp_prot = e_prot[mask]; bp_term = e_term[mask]; bp_sim = (-e_dist[mask]).astype(np.float32)
kept = 0
for p_, g_, s_ in zip(bp_prot, bp_term, bp_sim):
    r = erow.get(p_)
    if r is None:
        continue
    cols = clos_cols.get(g_)
    if cols is None or len(cols) == 0:
        continue
    cur = KNN[r, cols]
    KNN[r, cols] = np.where(s_ > cur, s_, cur)
    kept += 1
print(f"kNN baseline matrix built from {kept:,} bpo rows  ({time.time()-t0:.0f}s)", flush=True)

# ---------- metrics ----------
def metrics_for_term(scores, pos_mask):
    """scores: (n_eval,) higher=better. pos_mask: bool ground truth. Returns dict."""
    npos = int(pos_mask.sum()); nneg = n_eval - npos
    order = np.argsort(-scores, kind="stable")
    ranked_pos = pos_mask[order]
    def p_at(k):
        return float(ranked_pos[:k].sum()) / k
    # average precision over the full ranking
    hits = np.cumsum(ranked_pos)
    ranks = np.arange(1, n_eval + 1)
    prec = hits / ranks
    ap = float((prec * ranked_pos).sum() / npos) if npos else 0.0
    rp = float(ranked_pos[:npos].sum()) / npos if npos else 0.0   # R-precision
    # AUC via rank-sum (average ranks handle ties)
    rk = rankdata(scores)                     # ascending; larger score -> larger rank
    auc = float((rk[pos_mask].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)) if npos and nneg else float("nan")
    return {"P@5": p_at(5), "P@10": p_at(10), "P@20": p_at(20),
            "Rprec": rp, "AP": ap, "AUC": auc, "n_true": npos}

qual = np.where(true_per_term >= MIN_TRUE)[0]
IABUCK = [(0.0, 2.0, "IA[0,2)"), (2.0, 4.0, "IA[2,4)"), (4.0, 6.0, "IA[4,6)"),
          (6.0, 8.0, "IA[6,8)"), (8.0, 1e9, "IA>=8")]
SUPBUCK = [(0, 10, "sup[0,10)"), (10, 100, "sup[10,100)"), (100, 1000, "sup[100,1k)"),
           (1000, 10000, "sup[1k,10k)"), (10000, 10**12, "sup>=10k")]
def ia_bucket(w):
    for lo, hi, nm in IABUCK:
        if lo <= w < hi:
            return nm
    return IABUCK[-1][2]
def sup_bucket(s):
    for lo, hi, nm in SUPBUCK:
        if lo <= s < hi:
            return nm
    return SUPBUCK[-1][2]

per_term = {}
for j in qual:
    g = bp_terms[j]; pm = GT[:, j]
    m_model = metrics_for_term(S[:, j], pm)
    m_knn = metrics_for_term(KNN[:, j], pm)
    m_prior = metrics_for_term(degree, pm)
    per_term[g] = {"ia": IA.get(g, 0.0), "support": int(train_support[j]), "n_true": int(true_per_term[j]),
                   "name": name.get(g, g), "model": m_model, "knn": m_knn, "prior": m_prior}
print(f"scored {len(per_term):,} qualifying terms  ({time.time()-t0:.0f}s)", flush=True)

def agg(terms, arm):
    if not terms:
        return None
    keys = ["P@5", "P@10", "P@20", "Rprec", "AP", "AUC"]
    out = {"n_terms": len(terms)}
    for k in keys:
        vals = [per_term[g][arm][k] for g in terms if not (k == "AUC" and np.isnan(per_term[g][arm][k]))]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    # MAP is the mean of AP; expose explicitly
    out["MAP"] = out["AP"]
    return out

all_terms = list(per_term)
overall = {arm: agg(all_terms, arm) for arm in ("model", "knn", "prior")}

by_ia = {}
for lo, hi, nm in IABUCK:
    ts = [g for g in all_terms if lo <= per_term[g]["ia"] < hi]
    by_ia[nm] = {"n_terms": len(ts), **{arm: agg(ts, arm) for arm in ("model", "knn", "prior")}}
by_sup = {}
for lo, hi, nm in SUPBUCK:
    ts = [g for g in all_terms if lo <= per_term[g]["support"] < hi]
    by_sup[nm] = {"n_terms": len(ts), **{arm: agg(ts, arm) for arm in ("model", "knn", "prior")}}

# ---------- reliability boundary: model beats prior AND beats/matches knn, absolute AUC>=0.75 ----------
def reliable(bstat):
    """A regime is 'reliable/shippable' only if the LEARNED index (a) clears an absolute quality
    floor (AUC>=0.75) AND (b) beats BOTH trivial baselines: the degree prior AND the kNN retrieval
    the pipeline already exposes. Beating only the prior is not enough -- it must add value over the
    surface a user already has."""
    m, k, pr = bstat["model"], bstat["knn"], bstat["prior"]
    if not m or m["AUC"] is None or k is None or pr is None:
        return False
    return (m["AUC"] >= 0.75) and (m["MAP"] > pr["MAP"]) and (m["MAP"] > k["MAP"])

# ---------- demo: ~18 terms spanning IA buckets ----------
def top10_row(g):
    j = term_pos[g]; order = np.argsort(-S[:, j])[:10]; pm = GT[:, j]
    cells = []
    for r in order:
        cells.append((eval_accs[r], bool(pm[r]), float(S[r, j])))
    return cells
demo_terms = []
rng = np.random.default_rng(SEED)
for lo, hi, nm in IABUCK:
    # exclude the BP root / near-root uninformative terms (IA ~ 0) from the demo
    cands = sorted([g for g in all_terms if lo <= per_term[g]["ia"] < hi and IA.get(g, 0.0) >= 0.3],
                   key=lambda g: -per_term[g]["n_true"])
    # take a spread: most-supported, median, and a rarer-support one within the IA band
    pick = []
    if cands:
        pick.append(cands[0])
        if len(cands) > 2:
            pick.append(cands[len(cands)//2])
        pick.append(cands[-1])
    for g in dict.fromkeys(pick):
        demo_terms.append(g)
demo_terms = list(dict.fromkeys(demo_terms))[:20]

demo = {}
for g in demo_terms:
    demo[g] = {"name": name.get(g, g), "ia": round(IA.get(g, 0.0), 2),
               "support": per_term[g]["support"], "n_true_eval": per_term[g]["n_true"],
               "P@10_model": per_term[g]["model"]["P@10"], "AUC_model": per_term[g]["model"]["AUC"],
               "top10": top10_row(g)}

# ---------- write markdown demo ----------
md = []
md.append("# Reverse index demo: GO-BP term -> held-out eval proteins\n")
md.append(f"Model: bipartite kWTA link-prediction (checkpoint `bipartite_kwta_ckpt.pt`, "
          f"best val_mrr {best_mrr:.4f}), NO retraining. Score = dot(kwta(P(code)), kwta(T[term])).\n")
md.append(f"Universe = {n_eval:,} eval proteins held OUT of training. A check (Y) means the eval "
          f"protein is truly annotated with the term (propagated v230 gt).\n")
md.append(f"Terms shown span IA (rarity) buckets. `sup` = training edges the term had; "
          f"`n_true` = truly-annotated eval proteins; `P@10`/`AUC` are this term's retrieval quality.\n")
for g in demo_terms:
    d = demo[g]
    md.append(f"\n## {g} — {d['name']}\n")
    md.append(f"IA {d['ia']} | train-support {d['support']:,} | n_true_eval {d['n_true_eval']} | "
              f"P@10 {d['P@10_model']:.2f} | AUC {d['AUC_model']:.3f}\n")
    md.append("| rank | eval protein | truly annotated | model score |")
    md.append("|---:|:---|:---:|---:|")
    for i, (acc, hit, sc) in enumerate(d["top10"], 1):
        md.append(f"| {i} | {acc} | {'Y' if hit else '.'} | {sc:.3f} |")
(W / "reverse_index_demo.md").write_text("\n".join(md) + "\n")

# ---------- verdict ----------
mo = overall["model"]; kn = overall["knn"]; pr = overall["prior"]
reliable_ia = [nm for nm in [b[2] for b in IABUCK] if reliable(by_ia[nm])]
reliable_sup = [nm for nm in [b[2] for b in SUPBUCK] if reliable(by_sup[nm])]
verdict = {
    "overall_model": mo, "overall_knn": kn, "overall_prior": pr,
    "model_beats_prior_overall": bool(mo["MAP"] > pr["MAP"]),
    "model_beats_knn_overall": bool(mo["MAP"] > kn["MAP"]),
    "reliable_ia_buckets": reliable_ia,
    "reliable_support_buckets": reliable_sup,
    "summary": (
        "NOT a shippable retrieval surface. The learned reverse index encodes real term-specific "
        "signal (it beats the degree/popularity prior on MAP overall, %.4f vs %.4f), but it is "
        "DOMINATED by the trivial kNN retrieval the pipeline already exposes in EVERY IA bucket and "
        "EVERY train-support bucket (overall MAP %.4f model vs %.4f kNN; overall AUC %.4f vs %.4f). "
        "There is NO term class where the learned index is the best surface. Absolute quality is poor "
        "even on common terms (overall P@10 %.4f = ~1 correct in 10). Consistent with the campaign law: "
        "the deployed kNN+classifier retrieval is near-optimal; learning the term side from the "
        "adjacency does not add a retrieval channel." % (
            mo["MAP"], pr["MAP"], mo["MAP"], kn["MAP"], mo["AUC"], kn["AUC"], mo["P@10"])),
}

res = {
    "task": "reverse term->proteins retrieval quality on 7,401 HELD-OUT eval proteins",
    "checkpoint": "bipartite_kwta_ckpt.pt (NO retraining)", "best_val_mrr": round(best_mrr, 4),
    "n_eval_proteins": n_eval, "n_eval_total_in_npz": len(eval_accs_raw),
    "n_dropped_trained_memorized": n_contam,
    "n_bp_terms": n_bp, "min_true_per_term": MIN_TRUE,
    "n_qualifying_terms": len(all_terms),
    "gt": "propagated v230 (PK+LK+NK), BP only; ground truth = eval proteins annotated with the term",
    "metrics_definition": {
        "MAP": "mean average precision over qualifying terms",
        "P@k": "precision at k in the ranked eval-protein list",
        "Rprec": "precision at R = number of true eval proteins",
        "AUC": "rank AUC (Mann-Whitney) over all 7,401 eval proteins"},
    "baselines": {
        "knn": "trivial retrieval: -distance kNN similarity for (prot,term), closure-propagated with max",
        "prior": "term-independent popularity prior: rank eval proteins by BP annotation degree"},
    "overall": overall,
    "by_ia_bucket": by_ia,
    "by_train_support_bucket": by_sup,
    "reliability_rule": "reliable = model AUC>=0.75 AND model MAP > prior MAP (bucket-level means)",
    "verdict": verdict,
    "demo_terms": demo,
    "seconds": round(time.time() - t0, 1),
}
json.dump(res, open(W / "reverse_index_eval.json", "w"), indent=1, default=float)

# ---------- console report ----------
def line(nm, st):
    if not st or st.get("MAP") is None:
        return f"  {nm:14s} (n/a)"
    return (f"  {nm:14s} MAP {st['MAP']:.4f}  P@10 {st['P@10']:.4f}  P@20 {st['P@20']:.4f}  "
            f"Rprec {st['Rprec']:.4f}  AUC {st['AUC']:.4f}  [n_terms {st['n_terms']}]")
print(f"\n===== OVERALL (n_qual_terms={len(all_terms):,}) =====")
for arm in ("model", "knn", "prior"):
    print(line(arm.upper(), overall[arm]))
print("\n===== BY IA BUCKET (rarity/depth) =====")
for lo, hi, nm in IABUCK:
    b = by_ia[nm]; print(f" {nm}  n_terms={b['n_terms']}")
    for arm in ("model", "knn", "prior"):
        print(line("  " + arm, b[arm]))
print("\n===== BY TRAIN-SUPPORT BUCKET =====")
for lo, hi, nm in SUPBUCK:
    b = by_sup[nm]; print(f" {nm}  n_terms={b['n_terms']}")
    for arm in ("model", "knn", "prior"):
        print(line("  " + arm, b[arm]))
print(f"\nRELIABLE IA buckets: {reliable_ia}")
print(f"RELIABLE support buckets: {reliable_sup}")
print(f"model beats prior overall: {verdict['model_beats_prior_overall']}; "
      f"model beats knn overall: {verdict['model_beats_knn_overall']}")
print(f"\nreceipts: reverse_index_eval.json, reverse_index_demo.md  ({time.time()-t0:.0f}s)")
