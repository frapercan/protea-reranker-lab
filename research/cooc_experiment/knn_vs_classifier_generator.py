"""kNN or the classifier: which one should be GENERATING the candidates? Matched budget.

THE QUESTION NOBODY ASKED. The kNN has been the candidate generator since the beginning, and the
classifier has only ever been a supplement: we ask it for extras to top up a pool the kNN built.
The author's question is whether it should be the other way round. Nothing in this campaign has ever
put them in the same race.

`fullgo_ceiling.py` gives the reason to ask: the classifier's candidates are true **11.59%** of the
time at top5 against the co-occurrence expansion's **1.85%** and the DAG descent's **0.19%**. And of
the IA weight we miss, only **1.4%** falls outside its vocabulary. The information is there. The pool
never asks for it.

THE COMPARISON, at matched budget. The pk-bpo pool holds a median of ~140 candidates per protein, so
the classifier gets the same number and not one more. Anything else compares a generator to a
generator plus a bigger budget, which is two variables.

  KNN      the deployed pool exactly as it is
  CLF      the classifier's top-N BP terms per protein, N = that protein's own pool size
  UNION    both, to see whether they are finding the same truth or different truth

For each, per protein and IA-weighted over the propagated ground truth:
  RECALL   the share of the protein's true weight the generator makes reachable
  ORACLE   f_micro_w when a perfect ranker orders that generator's candidates. The ceiling.
  OVERLAP  how much of each one's truth the other already had. Two generators that find the same
           terms are one generator; the interesting case is disjoint truth.

WHAT THIS CANNOT SAY. Recall and a ceiling are not a system: the DAG descent had an oracle of 0.79
and a real arm of 0.0218 because `prop=fill` taxes every false candidate submitted. This measures
**generation**, and generation is only half the question. But the pool's own oracle is 0.7764 while
the deployed recipe delivers 0.2131, so if the classifier's ceiling is materially higher than the
kNN's at the same budget, the pool is the wrong shortlist and every ranking result on it inherits
that.

NO GATE, because this is a map and not a lever. The number to watch is the ORACLE at matched budget:
the pool's is 0.7764 by an independent route (`instrument_oracle_greedy.py`), so if CLF lands near or
above it, the generator is at least as good as the one we run and the campaign has been ranking the
wrong list. If it lands far below, the kNN is doing its job and the classifier is exactly what we
have been treating it as: a supplement.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np, yaml, psycopg2, pyarrow.parquet as pq, torch, torch.nn as nn

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
ORDER = [("08234f06-ba76-4d7d-aaec-ae601096b4fa", 768), ("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560),
         ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536), ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280),
         ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152), ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass
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
prots = sorted(set(GT) & set(pool))
sizes = np.array([len(pool[p]) for p in prots])
print(f"pk-bpo: {len(prots):,} proteins | pool size per protein: median {int(np.median(sizes))}, "
      f"mean {sizes.mean():.0f}, total {sizes.sum():,}", flush=True)

cfg = yaml.safe_load(Path("/home/frapercan/Thesis2/repositories/PROTEA/protea/config/system.yaml").read_text())
url = None
def find(d):
    global url
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, str) and v.startswith("postgresql"):
                url = v
            else:
                find(v)
    elif isinstance(d, list):
        for v in d:
            find(v)
find(cfg)
url = url.replace("postgresql+psycopg://", "postgresql://")
conn = psycopg2.connect(url); conn.set_session(readonly=True)
cur = conn.cursor()
t0 = time.time()
X = np.zeros((len(prots), 8320), np.float32)
pos = {a_: i for i, a_ in enumerate(prots)}
off = 0
for cid, dim in ORDER:
    for s in range(0, len(prots), 4000):
        ch = prots[s:s + 4000]
        cur.execute("""select p.accession, e.embedding::text from protein p
                       join sequence_embedding e on e.sequence_id = p.sequence_id
                       where e.embedding_config_id = %s and p.accession = any(%s)""", (cid, ch))
        for acc, emb in cur.fetchall():
            v = np.fromstring(emb.strip("[]"), sep=",", dtype=np.float32)
            if v.shape[0] == dim and acc in pos:
                X[pos[acc], off:off + dim] = v
    off += dim
conn.close()
print(f"  frame {X.shape} coverage {(np.abs(X).sum(1) > 0).mean():.1%}  ({time.time()-t0:.0f}s)", flush=True)

ck = torch.load("/home/frapercan/Thesis2/storage/fullgo_models/classifier_6plm_asl.pt",
                map_location="cpu", weights_only=False)
vocab = [alt.get(g, g) for g in ck["vocab"]]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
net = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, len(vocab)))
miss, _ = net.load_state_dict({k[4:] if k.startswith("net.") else k: v
                               for k, v in ck["state_dict"].items()}, strict=False)
assert not miss, f"partially loaded model: {miss}"
net = net.to(DEV).eval()
Xn = torch.tensor((X - ck["mu"].numpy()) / ck["sd"].numpy(), dtype=torch.float32)
with torch.no_grad():
    L = np.vstack([net(Xn[i:i + 256].to(DEV)).cpu().numpy() for i in range(0, len(Xn), 256)])
vb = np.array([i for i, g in enumerate(vocab) if g in BP])
vbg = [vocab[i] for i in vb]
LB = L[:, vb]
print(f"  logits {L.shape}; BP terms {len(vb):,}  ({time.time()-t0:.0f}s)", flush=True)

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''


def oracle(cands):
    """f_micro_w when a perfect ranker orders THIS generator's candidates: submit its true cells."""
    rows = []
    for p_, cs in cands.items():
        clos = set()
        for x in GT[p_]:
            clos |= {x} | anc(x)
        for g in cs:
            if g in clos:
                rows.append((p_, g, 1.0))
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows:
                fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, ts in GT.items():
                for g_ in ts:
                    fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA_F, toi=TOI, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


def reach(cands):
    """IA weight of the truth this generator makes reachable, over the truth's total."""
    got = tot = 0.0
    for p_ in prots:
        clos = set()
        for x in GT[p_]:
            clos |= {x} | anc(x)
        clos &= BP
        tot += sum(IA.get(x, 0.0) for x in clos)
        cl2 = set()
        for g in cands.get(p_, ()):
            cl2 |= {g} | anc(g)
        got += sum(IA.get(x, 0.0) for x in clos & cl2)
    return got, tot


knn = {p: set(pool[p]) for p in prots}
clf = {}
for i, p_ in enumerate(prots):
    n = len(pool[p_])                       # matched budget: this protein's own pool size
    top = np.argpartition(-LB[i], min(n, len(vb) - 1))[:n]
    clf[p_] = {vbg[j] for j in top}
uni = {p: knn[p] | clf[p] for p in prots}

res = {"question": "at matched budget, is the classifier a better candidate GENERATOR than the kNN?",
       "budget": "each protein gets exactly as many classifier candidates as its own pool holds",
       "pool_oracle_independent_route": 0.7764, "deployed_delivers": 0.2131, "arms": {}}
for tag, cands in (("KNN_the_deployed_pool", knn), ("CLF_matched_budget", clf), ("UNION", uni)):
    got, tot = reach(cands)
    o = oracle(cands)
    res["arms"][tag] = {"candidates": int(sum(len(v) for v in cands.values())),
                        "reachable_gt_ia": round(got, 1), "total_gt_ia": round(tot, 1),
                        "recall_of_gt_weight": round(got / tot, 4), "ORACLE_f_micro_w": o}
    r = res["arms"][tag]
    print(f"  {tag:24s} cands {r['candidates']:>9,}  reach {r['recall_of_gt_weight']:6.1%}  ORACLE {o}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "knn_vs_classifier_generator.json", "w"), indent=1)

# are they finding the same truth or different truth?
kt = ct = both = 0.0
for p_ in prots:
    clos = set()
    for x in GT[p_]:
        clos |= {x} | anc(x)
    clos &= BP
    for x in clos:
        w = IA.get(x, 0.0)
        ink = any(x == g or x in anc(g) for g in knn[p_])
        inc = any(x == g or x in anc(g) for g in clf[p_])
        if ink and inc:
            both += w
        elif ink:
            kt += w
        elif inc:
            ct += w
res["disjointness"] = {"only_the_knn_reaches_it": round(kt, 1),
                       "only_the_classifier_reaches_it": round(ct, 1),
                       "both": round(both, 1)}
json.dump(res, open(W / "knn_vs_classifier_generator.json", "w"), indent=1)
print(f"\n=== do they find the SAME truth? (IA weight) ===", flush=True)
print(f"  only the kNN reaches it:        {kt:>9,.0f}", flush=True)
print(f"  only the classifier reaches it: {ct:>9,.0f}", flush=True)
print(f"  both:                           {both:>9,.0f}", flush=True)
print(f"\n  the pool's oracle by an independent route is 0.7764; the deployed recipe delivers 0.2131.", flush=True)
print("DONE", flush=True)
