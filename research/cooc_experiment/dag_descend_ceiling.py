"""The ceiling of descending the DAG from what we already predict.

WHERE THIS COMES FROM. `curate_bp_errors.py` opened the predictor and found that **94.0% of the
missed IA weight on PK-BP is "too shallow"**: the true term hangs below something we already
predicted. Only 6.0% is in a different branch. And of that 94%, **80.8% was never retrieved at all**
(31,110 IA weight over 24,831 terms), while 19.2% sat in the pool below the threshold.

So the shape of our failure is: we name the right process and never ask for the specific one. And
the evaluator's own asymmetry makes that expensive in exactly one direction. `prop=fill` propagates
UP and hands us every ancestor free; nothing propagates DOWN. **The ancestors are free and the
descendants never come, and the descendants are where the weight is.**

WHY THIS IS NOT THE CO-OCCURRENCE EXPANSION THE THESIS ALREADY KILLED. That one expanded by
statistical association: it lifted recall 0.322 to 0.480 and converted to **+0.002**, because the
candidates it added were true only **1.85%** of the time. This expands by **structure**, from terms
the model already scored above its own threshold, downward along paths it already believes. Whether
that is a different thing or the same failure wearing a different coat is the measurement.

WHAT IS MEASURED, per descent depth k = 1, 2, 3, unlimited:
  - candidates added, and **what fraction of them are true**: the number to beat is 1.85%.
  - gt IA weight newly reachable, against the 31,110 the shallow miss is worth.
  - the ORACLE f_micro_w on the expanded pool: submit every expanded cell that is in the propagated
    truth at 1.0. That is the ceiling of a perfect ranker over the bigger shortlist, and it is an
    upper bound nobody can reach.
  - the NAIVE f_micro_w: give every expanded candidate its parent's score. That is what a system
    with no new evidence can actually do, and last night's mechanism predicts it should HURT,
    because a submitted weak candidate blocks its own ancestor's free inheritance.

THE HONEST TENSION, stated before the numbers: this lever runs directly against the one we already
have. Withholding weak candidates is worth +0.0092 because unsubmitted cells inherit. Descending
adds candidates, and every false one forfeits the inheritance of the ancestor above it. **The
oracle can only go up when the pool grows; the naive arm is the one that says whether this is real.**

NO GATE on the oracle, because a ceiling cannot pass or fail: it bounds. The gate is on the naive
arm, and it is the same 0.0034 noise floor: if giving the descendants their parent's score does not
clear it, then structure alone does not carry specificity and the lever needs evidence we do not have.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
TAU = 0.393

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
ch = collections.defaultdict(set)
for c, ps in par.items():
    for p in ps:
        if c in BP and p in BP:
            ch[p].add(c)
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
P = np.asarray(t.column("protein_accession").to_pylist())[m]
G = np.asarray(t.column("go_term_id").to_pylist())[m]
S = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
pool = collections.defaultdict(dict)
for p_, g_, s_ in zip(P, G, S):
    pool[p_][alt.get(g_, g_)] = s_
print(f"pk-bpo pool: {len(P):,} rows over {len(pool):,} proteins; deployed tau={TAU}", flush=True)

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


t0 = time.time()
res = {"tau": TAU, "context": "94.0% of the missed IA weight is 'too shallow'; 80.8% of that was "
                              "never retrieved (31,110 IA over 24,831 terms). fill propagates UP "
                              "only, so descendants never come.",
       "number_to_beat": "the co-occurrence expansion the thesis reports: recall 0.322 -> 0.480, "
                         "candidates true 1.85% of the time, converts to +0.002.",
       "depths": {}}

for K in (1, 2, 3, 99):
    added_true = added_tot = 0
    new_w = 0.0
    orc, naive = [], []
    for prot, pl in pool.items():
        truth = GT.get(prot, set())
        said = {g for g, s in pl.items() if s > TAU}
        # descend K levels from what we already predicted
        exp, frontier = {}, {g: pl[g] for g in said}
        for _ in range(K):
            nxt = {}
            for g, sc in frontier.items():
                for kid in ch.get(g, ()):
                    if kid not in pl and kid not in exp:
                        nxt[kid] = sc            # naive: inherit the parent's score
            if not nxt:
                break
            exp.update(nxt); frontier = nxt
        clos_truth = set()
        for x in truth:
            clos_truth |= {x} | anc(x)
        for g in exp:
            added_tot += 1
            if g in clos_truth:
                added_true += 1
                if g not in pl:
                    new_w += IA.get(g, 0.0)
        for g, sc in list(pl.items()) + list(exp.items()):
            if g in clos_truth:
                orc.append((prot, g, 1.0))
            if sc > TAU:
                naive.append((prot, g, sc))
    fo, fn_ = cafa(orc), cafa(naive)
    res["depths"][f"k={K}"] = {
        "candidates_added": added_tot,
        "of_them_true": added_true,
        "precision_of_the_addition": round(added_true / added_tot, 4) if added_tot else None,
        "new_gt_ia_weight_reached": round(new_w, 1),
        "oracle_f_micro_w_on_the_expanded_pool": fo,
        "naive_f_micro_w_parent_score": fn_,
    }
    r = res["depths"][f"k={K}"]
    print(f"  descend k={K:<2}: added {added_tot:>9,} candidates, {added_true:>6,} true "
          f"({r['precision_of_the_addition']:.2%})  new gt IA {new_w:>8,.0f}  "
          f"ORACLE {fo}  naive {fn_}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(W / "dag_descend_ceiling.json", "w"), indent=1)

print(f"\n=== the ceiling of descending the DAG ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  for reference: the pool's own oracle is 0.7764 and the deployed recipe delivers 0.2131.", flush=True)
print(f"  the co-occurrence expansion the thesis kills added candidates true 1.85% of the time.", flush=True)
print("  ORACLE tells you the ceiling; NAIVE tells you whether structure alone carries specificity.", flush=True)
print("DONE", flush=True)
