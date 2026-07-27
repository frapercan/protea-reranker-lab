"""Does the generator improve if its head spends all its capacity on BP?

WHERE THIS SITS. Rung 2 of task #46. Rung 1 is answered and it is a NEGATIVE
(`generator_learned_codes.json`): the learned k-WTA codes LOSE to the six raw PLMs at every cut
(-1.70 / -1.13 / -1.07 points), so the representation is not what holds the 11.59% floor. The input
stays RAW. What changes here is the OUTPUT.

THE QUESTION. `classifier_6plm_asl` emits 29,461 logits across three aspects to reach the 19,240 BP
ones we actually consume: two thirds of its head, its loss and its gradient go to MF and CC terms
this arm never reads. A BP-only head spends all of it on BP. Does that buy candidate precision?

ONE VARIABLE. Same corpus (88,212), same v227-experimental propagated labels, same RAW 8320-d input,
same architecture, loss, schedule and seed. Only the output vocabulary changes:

  ALL   29,999 terms, three aspects   what the shipped model does
  BP    the BP terms only             all capacity on the aspect we read

THE INSTRUMENT, and it is the lesson rung 1 paid for. **Precision is not comparable across arms at
the same k, only at the same VOLUME.** My rung-1 RAW arm looked 6.5 points better than the shipped
model at top5 purely because it proposed 0.37x the candidates there; at top50, where the volumes
agreed to 0.6%, the precisions agreed to 0.32 points. So each arm is swept over k and reported as a
precision-vs-volume curve, and the arms are compared where their volumes MATCH.

THE PRECONDITION: the ALL arm must reproduce rung 1's RAW arm exactly (same seed, same data, same
code path). Not approximately: exactly. If it does not, this script is not running the experiment I
think it is, and the BP arm's number would be measuring the drift.

THE GATE, quantity named: candidate precision at the volume matched to the ALL arm's top50 operating
point (~75,900 extras, where the shipped model sits at 6.48%). BP minus ALL, positive to win.

WHAT THIS CANNOT SAY: candidate precision is not f_micro_w. `dag_descend_ceiling.py` is the standing
warning, an oracle of 0.79 with a real arm of 0.0218. If BP wins here, the next run is the full lever
(`bootstrap_the_lever.py` shape, paired bootstrap CI) with the new generator, and only that counts.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn
from scipy.sparse import load_npz

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
FR = W / "generator_frames"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
SEED = 42
KS = (5, 10, 20, 35, 50, 75, 100, 150)
RUNG1 = {"top5": 0.1805, "top20": 0.1031, "top50": 0.0680}   # the ALL arm must reproduce these
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

par = collections.defaultdict(set); ns = {}; alt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("namespace: "):
        ns[cur_] = line[11:]
    elif cur_ and line.startswith("is_a: GO:"):
        par[cur_].add(line[6:].split(" ! ")[0].strip())
    elif cur_ and line.startswith("relationship: part_of GO:"):
        par[cur_].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur_ and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur_
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
    AC[t] = o
    return o

accs = json.load(open(FR / "accs.json"))
vocab = json.load(open(FR / "vocab.json"))
Y = load_npz(FR / "labels.npz").tocsr()
X = np.hstack([np.load(FR / f"raw_{n}.npy") for n in
               ("ankh_base", "esm2_3b", "ankh_large", "esm2_650m", "esmc_600m", "prott5")])
assert X.shape[1] == 8320, f"the six PLMs must concatenate to 8320, got {X.shape[1]}"
vb = np.array([i for i, g in enumerate(vocab) if g in BP])
print(f"cached: {len(accs):,} proteins, vocab {len(vocab):,} of which BP {len(vb):,} "
      f"({len(vb)/len(vocab):.1%}); RAW {X.shape}  ({time.time()-t0:.0f}s)", flush=True)

GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(alt.get(t_, t_))
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
c = np.asarray(t.column("category").to_pylist()); a = np.asarray(t.column("aspect").to_pylist())
m = (c == "pk") & (a == "bpo")
eP = np.asarray(t.column("protein_accession").to_pylist())[m]
eG = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
pool = collections.defaultdict(set)
for p_, g_ in zip(eP, eG):
    pool[p_].add(g_)
eprots = sorted(set(GT) & set(pool))
held = set(eprots)
tr_idx = np.array([i for i, a_ in enumerate(accs) if a_ not in held])
apos = {a_: i for i, a_ in enumerate(accs)}
idx = np.array([apos[p] for p in eprots if p in apos])
keep = [p for p in eprots if p in apos]
CLOS = {p: set().union(*[{x} | anc(x) for x in GT[p]]) for p in keep}
print(f"blind PK-BP proteins {len(keep):,}; training on {len(tr_idx):,}  ({time.time()-t0:.0f}s)",
      flush=True)


class Head(nn.Module):
    def __init__(self, d_in, n_out, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(h, n_out))

    def forward(self, x):
        return self.net(x)


def asl(logits, y, gn=4.0, gp=1.0, clip=0.05):
    p = torch.sigmoid(logits)
    pm = (p - clip).clamp(min=0)
    lp = y * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gp
    ln = (1 - y) * torch.log((1 - pm).clamp(min=1e-8)) * pm ** gn
    return -(lp + ln).sum() / y.shape[0]


def run(cols, tag):
    """cols: the output vocabulary's column indices into `vocab`. Everything else is identical."""
    torch.manual_seed(SEED)
    Yc = Y[:, cols]
    names = [vocab[i] for i in cols]
    mu, sd = X[tr_idx].mean(0, keepdims=True), X[tr_idx].std(0, keepdims=True) + 1e-6
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    mdl = Head(X.shape[1], len(cols)).to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    for e in range(8):
        mdl.train()
        perm = np.random.default_rng(SEED + e).permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            yb = torch.tensor(Yc[b].toarray(), device=DEV)
            opt.zero_grad()
            l = asl(mdl(Xn[b].to(DEV)), yb)
            l.backward(); opt.step(); tot += float(l.detach()) * len(b)
        print(f"  {tag} epoch {e} loss {tot/len(perm):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    mdl.eval()
    with torch.no_grad():
        L = np.vstack([mdl(Xn[idx[i:i + 256]].to(DEV)).cpu().numpy()
                       for i in range(0, len(idx), 256)])
    # score only BP columns: the arm we consume is BP, whatever the head also emits.
    bcols = np.array([j for j, g in enumerate(names) if g in BP])
    LB, bg = L[:, bcols], [names[j] for j in bcols]
    curve = []
    for K in KS:
        tot = true = 0
        for i, prot in enumerate(keep):
            pl = pool[prot]; clos = CLOS[prot]
            top = np.argpartition(-LB[i], min(K, len(bcols) - 1))[:K]
            for j in top:
                g = bg[j]
                if g in pl:
                    continue
                tot += 1
                true += g in clos
        curve.append({"k": K, "volume": tot, "true": true,
                      "precision": round(true / tot, 4) if tot else None})
        print(f"  {tag} top{K:<3}: volume {tot:>7,} true {true:>6,} = {true/tot:.2%}", flush=True)
    del mdl; torch.cuda.empty_cache()
    return curve


def at_volume(curve, V):
    """Precision at a target volume, interpolating cumulative-true against volume.

    Comparing two arms at the same k compares them at different operating points; that mistake is
    what voided rung 1's stated precondition. Volume is the axis they share.
    """
    vs = np.array([c["volume"] for c in curve], float)
    ts = np.array([c["true"] for c in curve], float)
    if V < vs.min() or V > vs.max():
        return None
    return round(float(np.interp(V, vs, ts) / V), 4)


res = {"question": "does a BP-only head beat a three-aspect head as a candidate generator?",
       "rung": "2 of task #46; rung 1 (the learned codes) is a NEGATIVE, so the input stays RAW.",
       "one_variable": "same 88,212 corpus, same v227-experimental propagated labels, same RAW "
                       "8320-d input, same architecture/loss/schedule/seed; only the output "
                       "vocabulary changes.",
       "instrument": "precision-vs-volume curves, compared at MATCHED volume. Precision is not "
                     "comparable across arms at the same k.",
       "arms": {}}
res["arms"]["ALL_three_aspects"] = run(np.arange(len(vocab)), "ALL")
res["arms"]["BP_only"] = run(vb, "BP")

A, B = res["arms"]["ALL_three_aspects"], res["arms"]["BP_only"]
byk = lambda c, k: next(x for x in c if x["k"] == k)
drift = {f"top{k}": round(byk(A, k)["precision"] - RUNG1[f"top{k}"], 4) for k in (5, 20, 50)}
ok = all(abs(v) < 0.002 for v in drift.values())
V = byk(A, 50)["volume"]
pa, pb = byk(A, 50)["precision"], at_volume(B, V)
res["precondition"] = {
    "requirement": "the ALL arm must reproduce rung 1's RAW arm exactly (same seed, same data).",
    "rung1_RAW": RUNG1, "this_run_ALL": {f"top{k}": byk(A, k)["precision"] for k in (5, 20, 50)},
    "drift": drift, "tolerance": 0.002, "PASSES": bool(ok)}
res["verdict"] = {
    "matched_volume": V,
    "ALL_precision_at_that_volume": pa, "BP_precision_at_that_volume": pb,
    "BP_minus_ALL": round(pb - pa, 4) if pb is not None else None,
    "BP_WINS": bool(ok and pb is not None and pb > pa),
    "VOID_IF_PRECONDITION_FAILS": not ok}
json.dump(res, open(W / "generator_bp_only_head.json", "w"), indent=1)
v = res["verdict"]
print(f"\n=== the generator's head ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION (ALL reproduces rung 1): {ok}   drift {drift}", flush=True)
print(f"  at the matched volume of {V:,} extras:", flush=True)
print(f"    ALL (29,999 terms, three aspects) {pa}", flush=True)
print(f"    BP  ({len(vb):,} terms, BP only)   {pb}", flush=True)
print(f"  BP - ALL = {v['BP_minus_ALL']}  -> BP-only head wins: {v['BP_WINS']}", flush=True)
print("  Precision is not f_micro_w. If BP wins, the next run is the full lever with it.", flush=True)
print("DONE", flush=True)
