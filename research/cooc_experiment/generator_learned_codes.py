"""Does the generator improve if you give it the learned codes instead of six raw PLMs?

WHERE THIS SITS. `THE_LEVER.md` records the campaign's only surviving lever: the full-GO
classifier's extra candidates, scored by a transferred model S, gain **+0.02245** with a paired
bootstrap CI of **[0.01791, 0.02899]**, 20 of 20 resamples positive. The generator that produces
them is `classifier_6plm_asl`, and the author's objection to it was correct and is the point: it is
old, unoptimised, and **it does not use the learned encoder**. Its input is 8320-d, six RAW PLMs
concatenated (ankh_base 768 + esm2_3b 2560 + ankh_large 1536 + esm2_650m 1280 + esmc_600m 1152 +
prott5 1024). `d8979601`, the learned k-WTA champion and the DEPLOYED encoder, is what the two-tower
gets and what the full-GO model has never been given.

So every number in THE_LEVER is a floor, and this is the first of the three ways left to raise it.
The other two are a BP-only head (it currently spends 29,461 outputs across three aspects to hit
19,240 BP ones) and a tuning sweep. **"More data" is not on the list**: the classifier already
trains on all 88,212 experimental proteins, which the task claiming otherwise asserted and the
builder disproved.

ONE VARIABLE. Same corpus (88,212 v227-experimental proteins), same labels (v227 experimental,
true-path propagated), same architecture, same loss, same schedule. Only the input changes.

  RAW      8320-d, the six concatenated PLMs      what the shipped model eats
  LEARNED  2048-d, the `d8979601` champion codes  what the deployed encoder produces

THE PRECONDITION, and it is the one that matters here: **the RAW arm's candidate precision must land
near the shipped model's 11.59% at top5.** I am training both arms myself, so if my RAW arm does not
reproduce the artefact it replaces, my training setup differs from theirs and the LEARNED arm's
number would be measuring my setup rather than the representation. Tolerance: 3 points absolute.
Outside it, the run is void and says nothing about the codes.

THE GATE, quantity named: candidate precision at top5 on the PK-BP blind window, LEARNED minus RAW.
The bar the whole line has to clear is the shipped model's **11.59%**, itself measured against
co-occurrence's 1.85% and DAG descent's 0.19%. Precision is the right quantity because the lever's
mechanism is precision: under `prop=fill` every false extra forfeits its ancestor's free
inheritance, which is why the raw-sigmoid arm loses at top50 with the very candidates S turns into
+0.0225.

LEAKAGE: labels come from `annotation_set c905dffa` = v227 = our t0, experimental codes only, which
is exactly what `extract_base_plm.py` does. The window v227-v230 is strictly after. The d8979601
codes are the deployed encoder's, trained before t0.

WHAT THIS CANNOT SAY: candidate precision is not f_micro_w. A better generator still has to survive
the scorer and the tax, and `dag_descend_ceiling.py` is the cautionary tale, an oracle of 0.79 with a
real arm of 0.0218. If LEARNED wins here, the next run is the full lever with the new generator, and
only that number counts.
"""
import json, collections, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"
EXP = ('EXP', 'IDA', 'IMP', 'IPI', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP')
CFG_LEARNED = "d8979601-ea59-4de1-9c16-21036ed67c36"
ORDER = [("08234f06-ba76-4d7d-aaec-ae601096b4fa", 768), ("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560),
         ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536), ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280),
         ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152), ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]
SEED = 42
SHIPPED_TOP5_PRECISION = 0.1159
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

# The frames and labels are cached by `cache_generator_frames.py` (146s, six PLMs at 100% coverage,
# one .npy per config). The earlier version re-read 88,212 x 6 PLMs from the DB on every attempt and
# died twice mid-pull with nothing in its log, which meant five minutes of setup to fail again. A run
# that cannot fail cheaply cannot be debugged.
from scipy.sparse import load_npz
FR = W / "generator_frames"
accs = json.load(open(FR / "accs.json"))
vocab = json.load(open(FR / "vocab.json"))
vpos = {g: i for i, g in enumerate(vocab)}
Y = load_npz(FR / "labels.npz")
print(f"cached: {len(accs):,} proteins, vocab {len(vocab):,}, labels {Y.shape} "
      f"density {Y.nnz/(Y.shape[0]*Y.shape[1]):.4%}  ({time.time()-t0:.0f}s)", flush=True)
XL = np.load(FR / "learned_2048.npy")
XR = np.hstack([np.load(FR / f"raw_{n}.npy") for n in
                ("ankh_base", "esm2_3b", "ankh_large", "esm2_650m", "esmc_600m", "prott5")])
assert XR.shape[1] == 8320, f"the six PLMs must concatenate to 8320, got {XR.shape[1]}"
print(f"  LEARNED {XL.shape} | RAW {XR.shape}  ({time.time()-t0:.0f}s)", flush=True)

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
# the blind window's proteins must be OUT of training: they are v227-v230 targets and the labels are
# v227 t0, but a protein can be both, so hold them out explicitly rather than trust the dates.
held = set(eprots)
tr_idx = np.array([i for i, a_ in enumerate(accs) if a_ not in held])
print(f"blind PK-BP proteins {len(eprots):,}; training on {len(tr_idx):,} of {len(accs):,} "
      f"(the blind proteins are held out explicitly)  ({time.time()-t0:.0f}s)", flush=True)


class Head(nn.Module):
    def __init__(self, d_in, n_out, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(h, n_out))

    def forward(self, x):
        return self.net(x)


def asl(logits, y, gn=4.0, gp=1.0, clip=0.05):
    """Asymmetric loss, as the shipped model uses: down-weight the easy negatives."""
    p = torch.sigmoid(logits)
    pm = (p - clip).clamp(min=0)
    lp = y * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gp
    ln = (1 - y) * torch.log((1 - pm).clamp(min=1e-8)) * pm ** gn
    return -(lp + ln).sum() / y.shape[0]


def train_and_measure(X, tag):
    mu, sd = X[tr_idx].mean(0, keepdims=True), X[tr_idx].std(0, keepdims=True) + 1e-6
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    mdl = Head(X.shape[1], len(vocab)).to(DEV)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    for e in range(8):
        mdl.train()
        perm = np.random.default_rng(SEED + e).permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            yb = torch.tensor(Y[b].toarray(), device=DEV)
            opt.zero_grad()
            l = asl(mdl(Xn[b].to(DEV)), yb)
            l.backward(); opt.step(); tot += float(l.detach()) * len(b)
        print(f"  {tag} epoch {e} loss {tot/len(perm):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    mdl.eval()
    vb = np.array([i for i, g in enumerate(vocab) if g in BP])
    vbg = [vocab[i] for i in vb]
    out = {}
    ei = np.array([accs.index(p) if p in vpos else -1 for p in eprots]) if False else None
    apos = {a_: i for i, a_ in enumerate(accs)}
    idx = np.array([apos[p] for p in eprots if p in apos])
    keep_prots = [p for p in eprots if p in apos]
    with torch.no_grad():
        L = np.vstack([mdl(Xn[idx[i:i + 256]].to(DEV)).cpu().numpy() for i in range(0, len(idx), 256)])
    LB = L[:, vb]
    for K in (5, 20, 50):
        tot = true = 0
        for i, prot in enumerate(keep_prots):
            clos = set()
            for x in GT[prot]:
                clos |= {x} | anc(x)
            pl = pool[prot]
            top = np.argpartition(-LB[i], min(K, len(vb) - 1))[:K]
            for j in top:
                g = vbg[j]
                if g in pl:
                    continue
                tot += 1
                if g in clos:
                    true += 1
        out[f"top{K}"] = {"added": tot, "true": true,
                          "precision": round(true / tot, 4) if tot else None}
        print(f"  {tag} top{K}: added {tot:>7,} true {true:>6,} = {out[f'top{K}']['precision']:.2%}", flush=True)
    del mdl; torch.cuda.empty_cache()
    return out


res = {"question": "does the generator improve with the learned codes instead of six raw PLMs?",
       "one_variable": "same 88,212-protein corpus, same v227-experimental propagated labels, same "
                       "architecture/loss/schedule; only the input representation changes.",
       "shipped_reference_top5_precision": SHIPPED_TOP5_PRECISION,
       "precondition": "the RAW arm must land within 3 points of the shipped 11.59%, else my "
                       "training setup differs from theirs and the comparison measures me, not the codes.",
       "bar": "the whole line's bar is 11.59%; co-occurrence was 1.85% and DAG descent 0.19%.",
       "n_train": int(len(tr_idx)), "vocab": len(vocab), "arms": {}}
res["arms"]["RAW_8320_six_plms"] = train_and_measure(XR, "RAW")
del XR
res["arms"]["LEARNED_2048_d8979601"] = train_and_measure(XL, "LEARNED")
json.dump(res, open(W / "generator_learned_codes.json", "w"), indent=1)

pr = res["arms"]["RAW_8320_six_plms"]["top5"]["precision"]
pl = res["arms"]["LEARNED_2048_d8979601"]["top5"]["precision"]
ok = pr is not None and abs(pr - SHIPPED_TOP5_PRECISION) < 0.03
res["verdict"] = {"RAW_top5": pr, "LEARNED_top5": pl,
                  "PRECONDITION_raw_reproduces_shipped": bool(ok),
                  "LEARNED_minus_RAW": round(pl - pr, 4) if (pr and pl) else None,
                  "LEARNED_WINS": bool(ok and pr and pl and pl > pr)}
json.dump(res, open(W / "generator_learned_codes.json", "w"), indent=1)
v = res["verdict"]
print(f"\n=== the generator's representation ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  PRECONDITION: RAW top5 {pr} vs the shipped {SHIPPED_TOP5_PRECISION} -> {ok}", flush=True)
print(f"  RAW      (8320-d, six raw PLMs)      top5 precision {pr}", flush=True)
print(f"  LEARNED  (2048-d, d8979601 champion) top5 precision {pl}", flush=True)
print(f"  LEARNED - RAW = {v['LEARNED_minus_RAW']}  -> learned codes win: {v['LEARNED_WINS']}", flush=True)
print("  Precision is not f_micro_w: a better generator still has to survive the scorer and the", flush=True)
print("  fill tax. If LEARNED wins, the next run is the full lever with it.", flush=True)
print("DONE", flush=True)
