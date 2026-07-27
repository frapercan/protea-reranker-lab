"""Rungs 1 and 2 of task #46, measured in IA rather than in counts.

WHY THIS SUPERSEDES `generator_learned_codes.py`. That script scored a candidate as "true" if it
appeared anywhere in the propagated ground-truth closure, and counted every hit as one. Both halves
are wrong, and the author caught it:

  1. **f_micro_w weights by IA.** A true candidate with IA near zero does not move the metric. Counted
     precision pays it the same as a specific one.
  2. **The closure runs up to the root.** "biological_process" is in every BP protein's closure, so
     proposing it scores as a hit. This is the exact shape of the vacuity that made "94% of the miss
     is too shallow" meaningless: a cut that everything passes measures nothing.
  3. **Under `prop=fill` a candidate is worth the IA it ADDS.** Predicting g fills g's ancestors, so
     if the pool already covers them, g brings only its own weight; and a FALSE g drags its false
     ancestors in with it. That asymmetry is the tax, and counting is blind to it.

THE INSTRUMENT. For each protein: the pool's closure P, the arm's added candidates' closure A. What
the arm actually contributes is A \\ P, weighted by IA:

    new_true_IA  = IA( (A \\ P) & gt_closure )      what the extras buy
    new_false_IA = IA( (A \\ P) - gt_closure )      what they cost
    IA precision = new_true_IA / (new_true_IA + new_false_IA)

Counted precision is kept alongside, only so the arms stay comparable to the shipped model's
published 11.59% / 9.84% / 6.48%.

MATCHED VOLUME, the lesson rung 1 paid for. Precision is not comparable across arms at the same k:
the first RAW arm looked 6.5 points better than the shipped model at top5 purely because it proposed
0.37x the candidates there, and at top50, where volumes agreed to 0.6%, the precisions agreed to 0.32
points. Arms are swept over k and compared where their candidate budgets MATCH.

THE ARMS, one variable each, off the same cached frames, same seed, schedule, loss and architecture:

    RAW_ALL      8320-d six raw PLMs -> 29,999 terms     the shipped model's shape (the control)
    LEARNED_ALL  2048-d d8979601     -> 29,999 terms     rung 1: is the representation the bottleneck?
    RAW_BP       8320-d six raw PLMs -> BP terms only    rung 2: is the head's aspect spread the cost?

THE PRECONDITION, on the IA axis this time, which is stronger than the counted one: the RAW_ALL
control must reproduce the shipped model's published `new_gt_ia_weight_reached` (270.8 / 1936.0 /
5715.5 at top5/20/50) at matched volume. If my control does not land near the artefact it stands in
for, my setup is not theirs and neither rung is answerable.

THE GATE: IA precision at the volume matched to RAW_ALL's top50 point (~75,900 extras). Arm minus
control, positive to win.

WHAT THIS CANNOT SAY: IA precision is still not f_micro_w. `dag_descend_ceiling.py` stands as the
warning (oracle 0.79, real arm 0.0218). A winner here earns a full-lever run, not a claim.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn
from scipy.sparse import load_npz

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
FR = W / "generator_frames"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO, IA_F = T0D / "go-basic.obo", T0D / "IA.tsv"   # the same pair fullgo_ceiling.py used
SEED = 42
KS = (5, 10, 20, 35, 50, 75, 100, 150)
SHIPPED = {5: {"volume": 1803, "count_prec": 0.1159, "new_gt_ia": 270.8},
           20: {"volume": 16444, "count_prec": 0.0984, "new_gt_ia": 1936.0},
           50: {"volume": 76348, "count_prec": 0.0648, "new_gt_ia": 5715.5}}
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

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2 and p[0].startswith("GO:"):
        try:
            IA[alt.get(p[0], p[0])] = float(p[1])
        except ValueError:
            pass
print(f"IA: {len(IA):,} terms, mean {np.mean(list(IA.values())):.3f}, "
      f"BP {sum(1 for g in IA if g in BP):,}", flush=True)
iaw = lambda S: float(sum(IA.get(g, 0.0) for g in S))

accs = json.load(open(FR / "accs.json"))
vocab = json.load(open(FR / "vocab.json"))
Y = load_npz(FR / "labels.npz").tocsr()
XR = np.hstack([np.load(FR / f"raw_{n}.npy") for n in
                ("ankh_base", "esm2_3b", "ankh_large", "esm2_650m", "esmc_600m", "prott5")])
assert XR.shape[1] == 8320, f"the six PLMs must concatenate to 8320, got {XR.shape[1]}"
XL = np.load(FR / "learned_2048.npy")
vb = np.array([i for i, g in enumerate(vocab) if g in BP])
print(f"cached: {len(accs):,} proteins, vocab {len(vocab):,} (BP {len(vb):,}); "
      f"RAW {XR.shape} LEARNED {XL.shape}  ({time.time()-t0:.0f}s)", flush=True)

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
# the two closures every measurement is relative to, computed once.
CLOS = {p: set().union(*[{x} | anc(x) for x in GT[p]]) for p in keep}          # the truth, propagated
PCLOS = {p: set().union(*[{x} | anc(x) for x in pool[p]]) for p in keep}       # what fill already gives us
print(f"blind PK-BP proteins {len(keep):,}; training on {len(tr_idx):,}; "
      f"mean pool closure {np.mean([len(v) for v in PCLOS.values()]):.0f} terms  "
      f"({time.time()-t0:.0f}s)", flush=True)


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


def run(X, cols, tag):
    torch.manual_seed(SEED)
    Yc, names = Y[:, cols], [vocab[i] for i in cols]
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
    bcols = np.array([j for j, g in enumerate(names) if g in BP])
    LB, bg = L[:, bcols], [names[j] for j in bcols]
    curve = []
    for K in KS:
        vol = hits = 0
        t_ia = f_ia = 0.0
        for i, prot in enumerate(keep):
            pl, clos, pc = pool[prot], CLOS[prot], PCLOS[prot]
            top = np.argpartition(-LB[i], min(K, len(bcols) - 1))[:K]
            added = {bg[j] for j in top} - pl
            vol += len(added)
            hits += sum(g in clos for g in added)
            # what fill actually gains from these extras: their closure MINUS what the pool's own
            # closure already covered. A true term whose ancestors we already had is nearly free of
            # gain; a false one drags its false ancestors along as cost.
            new = set().union(*[{g} | anc(g) for g in added]) - pc if added else set()
            t_ia += iaw(new & clos)
            f_ia += iaw(new - clos)
        curve.append({"k": K, "volume": vol, "count_hits": hits,
                      "count_precision": round(hits / vol, 4) if vol else None,
                      "new_true_ia": round(t_ia, 1), "new_false_ia": round(f_ia, 1),
                      "ia_precision": round(t_ia / (t_ia + f_ia), 4) if (t_ia + f_ia) else None})
        cc = curve[-1]
        print(f"  {tag} top{K:<3}: vol {vol:>7,} | count {cc['count_precision']:.2%} | "
              f"IA true {t_ia:>8.1f} false {f_ia:>9.1f} = {cc['ia_precision']:.2%}", flush=True)
    del mdl; torch.cuda.empty_cache()
    return curve


def at_volume(curve, V, field):
    """Interpolate a cumulative quantity against volume: arms share a budget axis, not a k axis."""
    vs = np.array([c["volume"] for c in curve], float)
    if V < vs.min() or V > vs.max():
        return None
    if field == "ia_precision":
        t = np.interp(V, vs, [c["new_true_ia"] for c in curve])
        f = np.interp(V, vs, [c["new_false_ia"] for c in curve])
        return round(float(t / (t + f)), 4) if (t + f) else None
    if field == "count_precision":
        return round(float(np.interp(V, vs, [c["count_hits"] for c in curve]) / V), 4)
    return round(float(np.interp(V, vs, [c[field] for c in curve])), 1)


res = {"question": "rungs 1 and 2 of #46, measured in IA instead of counts.",
       "why_remeasured": "the counted precision in generator_learned_codes.json scored a root-level "
                         "term as a hit and paid it like a specific one. f_micro_w weights by IA and "
                         "prop=fill pays a candidate for the IA it ADDS over the pool's closure.",
       "instrument": "new_true_ia / (new_true_ia + new_false_ia) over the arm's closure MINUS the "
                     "pool's closure; arms compared at MATCHED candidate volume.",
       "ia_table": str(IA_F), "obo": str(OBO), "arms": {}}
res["arms"]["RAW_ALL_control"] = run(XR, np.arange(len(vocab)), "RAW_ALL")
res["arms"]["LEARNED_ALL_rung1"] = run(XL, np.arange(len(vocab)), "LEARNED")
res["arms"]["RAW_BP_only_rung2"] = run(XR, vb, "RAW_BP")

C = res["arms"]["RAW_ALL_control"]
byk = lambda c, k: next(x for x in c if x["k"] == k)
pre = {}
for k, s in SHIPPED.items():
    mine = at_volume(C, s["volume"], "new_true_ia")
    pre[f"top{k}"] = {"shipped_volume": s["volume"], "shipped_new_gt_ia": s["new_gt_ia"],
                      "my_control_new_gt_ia_at_that_volume": mine,
                      "ratio": round(mine / s["new_gt_ia"], 3) if mine else None}
ratios = [v["ratio"] for v in pre.values() if v["ratio"]]
ok = len(ratios) == 3 and all(0.85 < r < 1.15 for r in ratios)
res["precondition"] = {
    "requirement": "the RAW_ALL control must reach the shipped model's published new_gt_ia_weight "
                   "(270.8 / 1936.0 / 5715.5) at MATCHED volume, within 15%.",
    "per_k": pre, "PASSES": bool(ok),
    "note": "this is the IA-axis precondition. The counted-precision one in "
            "generator_learned_codes.json was checked at top5, where the volumes differ 2.7x, and it "
            "read as a failure that was purely the operating point."}
V = byk(C, 50)["volume"]
res["verdict"] = {"matched_volume": V, "VOID_IF_PRECONDITION_FAILS": not ok, "arms": {}}
base = at_volume(C, V, "ia_precision")
for name, cur in res["arms"].items():
    p = at_volume(cur, V, "ia_precision")
    res["verdict"]["arms"][name] = {
        "ia_precision_at_matched_volume": p,
        "count_precision_at_matched_volume": at_volume(cur, V, "count_precision"),
        "new_true_ia": at_volume(cur, V, "new_true_ia"),
        "minus_control": round(p - base, 4) if p is not None else None,
        "WINS": bool(ok and p is not None and p > base)}
json.dump(res, open(W / "generator_rungs_ia.json", "w"), indent=1)
print(f"\n=== rungs 1 + 2, in IA ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION (control reaches the shipped model's gt IA at matched volume): {ok}", flush=True)
for k, v in pre.items():
    print(f"    {k:>6}: shipped {v['shipped_new_gt_ia']:>7.1f} vs mine "
          f"{v['my_control_new_gt_ia_at_that_volume']} -> ratio {v['ratio']}", flush=True)
print(f"  at the matched volume of {V:,} extras, IA precision:", flush=True)
for name, v in res["verdict"]["arms"].items():
    print(f"    {name:<22} {v['ia_precision_at_matched_volume']}  "
          f"(count {v['count_precision_at_matched_volume']}, true IA {v['new_true_ia']})  "
          f"vs control {v['minus_control']:+}  wins={v['WINS']}", flush=True)
print("  IA precision is still not f_micro_w. A winner earns a full-lever run, not a claim.", flush=True)
print("DONE", flush=True)
