"""What is the +0.02245 lever actually made of: new candidates, or rescored inheritance?

THE DOUBT, and it is the author's. The rungs-in-IA run showed the classifier's true extras carry
almost no MARGINAL IA: at top50 their new-true IA is 470 against 33,000 of new-false IA, because the
pool's closure already covers 2.1x the IA weight we submit (129 terms per protein -> 358 in closure).
If the true extras add no new IA, the lever's +0.02245 in f_micro_w cannot be coming from reaching
new true branches. So where is it coming from?

THE HYPOTHESIS, from reading the construction. `score_the_extras.py:355` keeps an extra when
`g not in pl`, where `pl` is the pool's SUBMITTED terms, not its closure. So an extra can be an
ANCESTOR of a submitted term: a cell that `prop=fill` was already filling by inheritance
(max over descendants). Submitting it explicitly with S's score REPLACES that inherited value with
our own. Under `fill` (overwrites only exactly-zero cells) that is the only way to touch an inherited
cell. So the lever may not be candidate generation at all: it may be RESCORING what fill already gave
us, with a scorer better than max-over-descendants.

THE TEST, one variable. Take arm B's exact submitted rows (`lever_submission_top50.tsv`, 38,173
tagged extras over 39,808 pool rows, the artefact `bootstrap_the_lever.py` froze). Partition the
extras by whether each sits INSIDE the pool's propagated closure:

  IN   g is an ancestor of a submitted pool term    fill already granted it; submitting = RESCORING
  OUT  g is outside the closure                      a genuinely new candidate

Then evaluate four submissions on the board's line, same harness as bootstrap, to the digit:

  A          pool only                     the anchor, must reproduce 0.22282
  A + IN     rescored inheritance only     the rescoring channel
  A + OUT    new candidates only           the generation channel
  B          A + all extras                must reproduce 0.24527 (the +0.02245)

The two deltas decompose the lever. If A+IN carries it, the lever is a RESCORING lever, it does not
need the classifier as a generator, and it is not confined to PK-BP: it is a claim about fill's
inheritance being improvable everywhere. If A+OUT carries it, generation is real after all and the
IA-marginal reading is missing something. Either answer sets the direction.

NOT a new number for the thesis: B reproduces the sealed +0.02245. This only asks what it is.
"""
import json, collections, subprocess, tempfile, time
from pathlib import Path
import numpy as np

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO, IA_F = str(T0D / "go-basic.obo"), str(T0D / "IA.tsv")
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
CACHE = W / "lever_submission_top50.tsv"
ANCHOR_A, ANCHOR_B = 0.22282, 0.24527
t0 = time.time()

par = collections.defaultdict(set); alt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("is_a: GO:"):
        par[cur_].add(line[6:].split(" ! ")[0].strip())
    elif cur_ and line.startswith("relationship: part_of GO:"):
        par[cur_].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur_ and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur_
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

pool_rows, extra_rows = [], []
pool_terms = collections.defaultdict(set)
for line in CACHE.open():
    p_, g_, v, tag = line.rstrip("\n").split("\t")
    g_ = alt.get(g_, g_)
    (pool_rows if tag == "pool" else extra_rows).append((p_, g_, float(v)))
    if tag == "pool":
        pool_terms[p_].add(g_)
# the pool's closure per protein: exactly the cells fill inherits into (submitted terms plus all
# their ancestors). An extra inside this set is one fill was already granting for free.
closure = {p: set().union(*[{g} | anc(g) for g in ts]) for p, ts in pool_terms.items()}
in_rows, out_rows = [], []
for p_, g_, v in extra_rows:
    (in_rows if g_ in closure.get(p_, ()) else out_rows).append((p_, g_, v))
print(f"pool {len(pool_rows):,} rows | extras {len(extra_rows):,} = "
      f"IN-closure {len(in_rows):,} (rescored inheritance) + "
      f"OUT-closure {len(out_rows):,} (new candidates)  ({time.time()-t0:.0f}s)", flush=True)

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


def cafa(rws):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rws:
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
            print(r.stderr[-800:], flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None


A = cafa(pool_rows)
B = cafa(pool_rows + extra_rows)
A_in = cafa(pool_rows + in_rows)
A_out = cafa(pool_rows + out_rows)
res = {"question": "is the +0.02245 lever new candidates or rescored inheritance?",
       "counts": {"pool": len(pool_rows), "extras": len(extra_rows),
                  "in_closure_rescored": len(in_rows), "out_closure_new": len(out_rows)},
       "A_pool_only": A, "B_all_extras": B,
       "A_plus_IN_rescored_inheritance": A_in, "A_plus_OUT_new_candidates": A_out,
       "precondition_A": bool(A is not None and abs(A - ANCHOR_A) < 0.002),
       "precondition_B": bool(B is not None and abs(B - ANCHOR_B) < 0.002)}
if A is not None:
    res["delta_IN"] = round((A_in - A), 5) if A_in is not None else None
    res["delta_OUT"] = round((A_out - A), 5) if A_out is not None else None
    res["delta_B"] = round((B - A), 5) if B is not None else None
ok = res["precondition_A"] and res["precondition_B"]
res["VOID_IF_PRECONDITION_FAILS"] = not ok
if ok and res.get("delta_IN") is not None and res.get("delta_OUT") is not None:
    res["VERDICT"] = ("RESCORING lever: rescored inheritance carries it"
                      if res["delta_IN"] > res["delta_OUT"] else
                      "GENERATION lever: new candidates carry it")
    res["additivity_note"] = (f"delta_IN {res['delta_IN']} + delta_OUT {res['delta_OUT']} vs "
                              f"delta_B {res['delta_B']}; fill makes the channels non-additive, so "
                              f"the split is diagnostic, not a decomposition to the digit.")
json.dump(res, open(W / "lever_mechanism_partition.json", "w"), indent=1)
print(f"\n=== what the lever is made of ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  A   pool only                         {A}  (anchor {ANCHOR_A}, ok {res['precondition_A']})",
      flush=True)
print(f"  B   + all {len(extra_rows):,} extras            {B}  (anchor {ANCHOR_B}, ok {res['precondition_B']})",
      flush=True)
print(f"  A + IN  ({len(in_rows):,} rescored inheritance)  {A_in}  delta {res.get('delta_IN')}",
      flush=True)
print(f"  A + OUT ({len(out_rows):,} new candidates)       {A_out}  delta {res.get('delta_OUT')}",
      flush=True)
print(f"  VERDICT: {res.get('VERDICT', 'VOID (precondition failed)')}", flush=True)
print("DONE", flush=True)
