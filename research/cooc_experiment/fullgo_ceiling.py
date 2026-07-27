"""The ceiling of the full-GO multi-label classifier on PK-BP. Read-only.

WHY. `curate_bp_errors.py`, once its own vacuous cut was fixed, said this: **71.0% of the missed IA
weight on PK-BP is neither below, above nor among the terms we predict** (34,464 IA over 27,661
terms). And of that, **81.0% (27,903 IA) sits in the full-GO classifier's vocabulary but never in
our candidate pool**, while only **1.4% falls outside its vocabulary entirely**. The evidence is not
missing. We never ask for it.

The pool comes from kNN. The classifier is used only as a candidate generator with a cut. Its
candidates already carry 78% of the positives, and its logit alone beat the deployed reranker by a
gated +0.0084. It has never been asked about the whole vocabulary.

WHAT THIS MODEL IS, and the author's three objections, all fair:
  - **Old**: trained 2026-06-14, and its checkpoint records only `mu`, `sd`, `in_dim`, `hidden`,
    `vocab`. **No training-set provenance.** That had to be reconstructed by reading the builder.
  - **No learned embeddings**: the input is six RAW PLMs concatenated, 8320-d = ankh_base 768 +
    esm2_3b 2560 + ankh_large 1536 + esm2_650m 1280 + esmc_600m 1152 + prott5 1024. It does NOT use
    `d8979601`, the learned k-WTA champion. If it wins, it wins with the worse hand.
  - **Unknown missing optimisations**: a plain 8320 -> 1024 -> 29,461 MLP with asymmetric loss.

LEAKAGE, checked rather than assumed: `dump_classifier_inputs.py` pins the labels to
`ANN = c905dffa... # v227 t0` and filters to experimental evidence codes. **v227 is our t0 and the
evaluation window is v227 to v230, so the labels are strictly before it.** Clean.

**THIS RUN CANNOT KILL THE MULTI-LABEL DIRECTION, ONLY THIS ARTEFACT.** That inverts every other
gate in this campaign and is stated before the number, not after. A failure here says this old,
unoptimised, wrong-embedding model fails; a success says the direction has headroom, because the
floor already clears the bar.

THE NUMBERS TO BEAT, both from expansions this campaign already measured:
  the co-occurrence expansion the thesis reports:  candidates true **1.85%** of the time, +0.002
  descending the DAG (`dag_descend_ceiling.py`):   candidates true **0.19%**, and the real arm
                                                   collapsed from 0.2131 to **0.0218**

TWO ARMS, because a ceiling alone means nothing. The DAG descent had an oracle of 0.79 and a real
arm of 0.0218:
  ORACLE   submit the expanded cells that are in the propagated truth at 1.0. An upper bound.
  REAL     submit the classifier's own logits above a threshold swept per arm. What it can do.

The mechanism says the real arm is at risk: every false candidate submitted forfeits its ancestor's
free inheritance under `prop=fill`, and this model offers 29,461 candidates per protein against the
pool's ~140.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np, yaml, psycopg2, pyarrow.parquet as pq, torch

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
TAU_POOL = 0.393

ORDER = [("08234f06-ba76-4d7d-aaec-ae601096b4fa", 768),    # ankh_base, the frame's base
         ("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560),   # esm2_3b
         ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536),   # ankh_large
         ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280),   # esm2_650m
         ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152),   # esmc_600m
         ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]   # prott5
assert sum(d for _, d in ORDER) == 8320

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass
par = collections.defaultdict(set); ns = {}; alt = {}
cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur = None
    elif line.startswith("id: GO:"):
        cur = line[4:]
    elif cur and line.startswith("namespace: "):
        ns[cur] = line[11:]
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
Pp = np.asarray(t.column("protein_accession").to_pylist())[m]
Gg = np.asarray(t.column("go_term_id").to_pylist())[m]
Ss = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
pool = collections.defaultdict(dict)
for p_, g_, s_ in zip(Pp, Gg, Ss):
    pool[p_][alt.get(g_, g_)] = s_
prots = sorted(set(GT) & set(pool))
print(f"PK-BP proteins with both truth and a pool: {len(prots):,}", flush=True)

# ---- build the 8320-d frame from the DB, read-only ------------------------------
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
cur_db = conn.cursor()
t0 = time.time()
X = np.zeros((len(prots), 8320), np.float32)
pos = {a_: i for i, a_ in enumerate(prots)}
off = 0
for cfg_id, dim in ORDER:
    got = 0
    for s in range(0, len(prots), 4000):
        chunk = prots[s:s + 4000]
        cur_db.execute("""select p.accession, e.embedding::text from protein p
                          join sequence_embedding e on e.sequence_id = p.sequence_id
                          where e.embedding_config_id = %s and p.accession = any(%s)""",
                       (cfg_id, chunk))
        for acc, emb in cur_db.fetchall():
            v = np.fromstring(emb.strip("[]"), sep=",", dtype=np.float32)
            if v.shape[0] == dim and acc in pos:
                X[pos[acc], off:off + dim] = v; got += 1
    print(f"  {cfg_id[:8]} dim {dim:>5}: {got:>6,}/{len(prots):,} proteins  ({time.time()-t0:.0f}s)", flush=True)
    off += dim
cov = (np.abs(X).sum(1) > 0).mean()
print(f"  frame {X.shape}, proteins with any signal: {cov:.1%}", flush=True)

ck = torch.load("/home/frapercan/Thesis2/storage/fullgo_models/classifier_6plm_asl.pt",
                map_location="cpu", weights_only=False)
vocab = list(ck["vocab"])
mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import torch.nn as nn
# net.1 and net.5 carry a weight and a bias and NO running statistics, so they are LayerNorm and
# not BatchNorm. The first attempt assumed BatchNorm and the strict-load guard caught it: the
# checkpoint's own shapes name the architecture, and guessing at it would have scored with two
# uninitialised normalisation layers and produced a number that meant nothing.
net = nn.Sequential(nn.Linear(8320, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(1024, len(vocab)))
# the checkpoint stores the Sequential under a `net.` attribute: net.0 / net.4 / net.8 are the
# three Linears (8320->1024, 1024->1024, 1024->29461) and the indices already line up, so the only
# difference is the prefix. Strip it rather than guess at the architecture.
sdk = {k[4:] if k.startswith("net.") else k: v for k, v in ck["state_dict"].items()}
missing, unexpected = net.load_state_dict(sdk, strict=False)
if missing or unexpected:
    print(f"  load_state_dict: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}", flush=True)
    if missing:
        raise SystemExit("refusing to score with a partially loaded model")
print("  checkpoint loaded, every parameter matched", flush=True)
net = net.to(DEV).eval()
Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
with torch.no_grad():
    L = np.vstack([net(Xn[i:i + 512].to(DEV)).cpu().numpy() for i in range(0, len(Xn), 512)])
print(f"  logits {L.shape}  ({time.time()-t0:.0f}s)", flush=True)

vb = [i for i, g in enumerate(vocab) if alt.get(g, g) in BP]
vbg = [alt.get(vocab[i], vocab[i]) for i in vb]
print(f"  BP terms in the classifier vocabulary: {len(vb):,} of {len(vocab):,}", flush=True)

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


def cafa(rows):
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


LB = L[:, vb]
res = {"model": "classifier_6plm_asl (2026-06-14): 8320 raw-PLM -> 1024 -> 29,461 labels, ASL. "
                "NOT the learned k-WTA champion; six raw PLMs concatenated.",
       "leakage": "labels pinned to annotation_set c905dffa = v227 t0, experimental codes only; "
                  "the evaluation window v227-v230 is strictly after. Clean.",
       "this_cannot_kill_the_direction": "a failure indicts THIS artefact, not multi-label DL. "
                                         "It is a floor: old, unoptimised, wrong embeddings.",
       "numbers_to_beat": {"cooccurrence_expansion_precision": 0.0185,
                           "dag_descent_precision": 0.0019, "dag_descent_real_arm": 0.0218},
       "vocab_bp_terms": len(vb), "arms": {}}

for TOPK in (5, 20, 50):
    added_tot = added_true = 0
    new_w = 0.0
    orc, real = [], []
    for i, prot in enumerate(prots):
        truth = GT[prot]
        clos = set()
        for x in truth:
            clos |= {x} | anc(x)
        pl = pool[prot]
        top = np.argpartition(-LB[i], min(TOPK, len(vb) - 1))[:TOPK]
        for j in top:
            g = vbg[j]
            if g in pl:
                continue
            added_tot += 1
            if g in clos:
                added_true += 1; new_w += IA.get(g, 0.0)
        for g, s_ in pl.items():
            if s_ > TAU_POOL:
                real.append((prot, g, float(s_)))
            if g in clos:
                orc.append((prot, g, 1.0))
        mx = float(LB[i].max()) if len(vb) else 1.0
        for j in top:
            g = vbg[j]
            if g not in pl:
                sc = 1 / (1 + np.exp(-LB[i, j]))
                real.append((prot, g, float(sc)))
                if g in clos:
                    orc.append((prot, g, 1.0))
    fo, fr = cafa(orc), cafa(real)
    res["arms"][f"top{TOPK}"] = {
        "candidates_added": added_tot, "of_them_true": added_true,
        "precision_of_the_addition": round(added_true / added_tot, 4) if added_tot else None,
        "new_gt_ia_weight_reached": round(new_w, 1),
        "ORACLE_on_the_expanded_pool": fo, "REAL_arm_pool_plus_classifier": fr}
    r = res["arms"][f"top{TOPK}"]
    print(f"  top{TOPK:<3}: added {added_tot:>8,}  true {added_true:>6,} ({r['precision_of_the_addition']:.2%})  "
          f"new gt IA {new_w:>8,.0f}  ORACLE {fo}  REAL {fr}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "fullgo_ceiling.json", "w"), indent=1)

print(f"\n=== the full-GO classifier's ceiling on PK-BP ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  bar to clear: co-occurrence added candidates true 1.85% of the time; DAG descent 0.19%.", flush=True)
print(f"  the deployed recipe delivers 0.2131 and the pool's own oracle is 0.7764.", flush=True)
print("DONE", flush=True)
