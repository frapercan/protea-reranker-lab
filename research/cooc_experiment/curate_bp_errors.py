"""Open the predictor. Where in the ontology do we actually fail on PK-BP?

WHY THIS EXISTS. Every measurement in this campaign is an aggregate: f_micro_w, a delta, a fold
standard deviation. Nobody has looked at a protein. We know the pool is worth 0.7764 to a perfect
ordering and we deliver 0.2131, and we have spent the campaign trying to close that with technique.
Technique is exhausted: the objective, the architecture, the candidates, the structure features and
the text signals are each measured and each dead or flat.

What has never been asked is **what kind of biological mistake the 0.55 gap is made of**.

THE QUESTION THE MECHANISM MAKES SHARP. `prop=fill` gives every ancestor of a submitted term for
free, so the two ways to be wrong are not equally expensive:

  **too shallow**  the true term is a DESCENDANT of something we predicted. We named the right
                   process and stopped short of the specific one. The fill already gave us the
                   ancestors, so this failure costs only the information accretion between our
                   depth and the truth's.
  **wrong branch** the true term is not below anything we said. We are in a different part of the
                   ontology altogether and the fill cannot rescue us.

Those demand opposite levers. Too shallow is a specificity problem: push the same prediction deeper
along a path we already believe. Wrong branch is a retrieval or evidence problem, and no amount of
re-ranking a shortlist that lacks the right region will fix it.

**This has never been separated. Both fail the same way in f_micro_w.**

WHAT IT MEASURES, per PK-BP protein at the deployed operating point:
  - TP / FP / FN under the true-path closure, so a term we implied by an ancestor counts as said.
  - For every FN: its relation to what we DID predict. Is it a descendant of a predicted term
    (too shallow), an ancestor (we overshot), a sibling under a shared parent (near miss), or
    unreachable from anything we said (wrong branch)? Distance to the closest thing we predicted.
  - The information-accretion weight of each class, because f_micro_w counts IA and not terms.
  - The same split by the top-level BP subtree, so "which biology do we lose" has an answer.

NO GATE, and that is deliberate. This is not a lever test; it is a map. It cannot pass or fail. What
it can do is say which lever is worth building, and the honest outcome is that it might say none.
"""
import json, collections, math
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
BP_ROOT = "GO:0008150"

# ---- the ontology -------------------------------------------------------------
parents = collections.defaultdict(set)
ns_of, name_of, alt = {}, {}, {}
cur = None
for line in OBO.open():
    line = line.rstrip("\n")
    if line == "[Term]":
        cur = None
    elif line.startswith("id: GO:"):
        cur = line[4:]
    elif cur and line.startswith("name: "):
        name_of[cur] = line[6:]
    elif cur and line.startswith("namespace: "):
        ns_of[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"):
        parents[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        parents[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur
BP = {t for t, n in ns_of.items() if n == "biological_process"}
children = collections.defaultdict(set)
for c, ps in parents.items():
    for p in ps:
        children[p].add(c)
print(f"ontology: {len(ns_of):,} terms, {len(BP):,} BP", flush=True)

IA = {}
for line in IA_F.open():
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass


def ancestors(t, cache={}):
    if t in cache:
        return cache[t]
    out, stack = set(), [t]
    while stack:
        x = stack.pop()
        for p in parents.get(x, ()):
            if p not in out:
                out.add(p); stack.append(p)
    cache[t] = out
    return out


def depth(t, cache={}):
    if t in cache:
        return cache[t]
    if t == BP_ROOT or not parents.get(t):
        cache[t] = 0
        return 0
    d = 1 + min((depth(p) for p in parents[t] if p in BP), default=-1)
    cache[t] = max(d, 0)
    return cache[t]


# top-level BP branches: the direct children of the root, the coarse "which biology" axis
TOP = sorted(children.get(BP_ROOT, set()))
top_of_cache = {}


def top_branches(t):
    if t in top_of_cache:
        return top_of_cache[t]
    a = ancestors(t) | {t}
    r = {x for x in a if x in children.get(BP_ROOT, set())}
    top_of_cache[t] = r
    return r


# ---- the truth and our prediction ---------------------------------------------
GT = collections.defaultdict(set)
with (REL / "groundtruth_PK.tsv").open() as fh:
    next(fh)
    for line in fh:
        p_, t_, a_ = line.rstrip("\n").split("\t")[:3]
        if a_ == "P":
            GT[p_].add(alt.get(t_, t_))

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
cat = np.asarray(t.column("category").to_pylist()); asp = np.asarray(t.column("aspect").to_pylist())
m = (cat == "pk") & (asp == "bpo")
P = np.asarray(t.column("protein_accession").to_pylist())[m]
G = np.asarray(t.column("go_term_id").to_pylist())[m]
S = t.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)[m]
TAU = 0.393        # the deployed operating point on this cell (anchor_deployed_recipe.py)
sub = collections.defaultdict(set)
for p_, g_, s_ in zip(P, G, S):
    if s_ > TAU:
        sub[p_].add(alt.get(g_, g_))
print(f"proteins with a submission at tau={TAU}: {len(sub):,} | gt proteins: {len(GT):,}", flush=True)

# ---- the map -------------------------------------------------------------------
cls = collections.Counter()
wt = collections.Counter()
dist_hist = collections.Counter()
by_top_fn = collections.Counter(); by_top_tp = collections.Counter()
examples = collections.defaultdict(list)
n_prot = 0

for prot, truth in GT.items():
    said = sub.get(prot, set())
    if not truth:
        continue
    n_prot += 1
    # true-path closure: saying a term says all its ancestors (this is what `fill` does for us)
    closure = set(said)
    for s_ in said:
        closure |= ancestors(s_)
    closure &= BP
    truth_bp = {x for x in truth if x in BP}
    tp = truth_bp & closure
    fn = truth_bp - closure
    for x in tp:
        cls["TP"] += 1; wt["TP"] += IA.get(x, 0.0)
        for b in top_branches(x):
            by_top_tp[b] += IA.get(x, 0.0)
    for x in fn:
        w = IA.get(x, 0.0)
        anc_x = ancestors(x)
        desc_of_said = bool(anc_x & closure)          # x sits BELOW something we said
        anc_of_said = bool({x} & set().union(*[ancestors(s_) for s_ in said]) if said else set())
        if desc_of_said:
            k = "FN_too_shallow"
        elif anc_of_said:
            k = "FN_we_overshot"
        elif said and (anc_x & set().union(*[ancestors(s_) | {s_} for s_ in said])):
            k = "FN_sibling_near_miss"
        else:
            k = "FN_wrong_branch"
        cls[k] += 1; wt[k] += w
        for b in top_branches(x):
            by_top_fn[b] += w
        if len(examples[k]) < 4 and w > 3:
            near = min((len(anc_x ^ ancestors(s_)) for s_ in said), default=-1)
            examples[k].append({"protein": prot, "term": x, "name": name_of.get(x, "?"),
                                "ia": round(w, 2), "depth": depth(x),
                                "closest_said_symdiff": near})

tot_fn_w = sum(v for k, v in wt.items() if k.startswith("FN"))
print(f"\n=== what the PK-BP failure is MADE OF ({n_prot:,} proteins, IA-weighted) ===", flush=True)
print(f"  {'class':24s} {'terms':>9s} {'IA weight':>12s} {'share of the miss':>18s}")
for k in ("TP", "FN_too_shallow", "FN_sibling_near_miss", "FN_we_overshot", "FN_wrong_branch"):
    share = f"{wt[k]/tot_fn_w:.1%}" if k.startswith("FN") and tot_fn_w else ""
    print(f"  {k:24s} {cls[k]:>9,} {wt[k]:>12,.0f} {share:>18s}", flush=True)

print(f"\n=== which biology do we lose? top-level BP branches by missed IA weight ===", flush=True)
rows = sorted(by_top_fn.items(), key=lambda x: -x[1])[:12]
for b, w in rows:
    got = by_top_tp.get(b, 0.0)
    cap = got / (got + w) if (got + w) else 0
    print(f"  {b} {name_of.get(b,'?')[:44]:44s} missed {w:>9,.0f}  captured {cap:5.1%}", flush=True)

out = {"tau": TAU, "n_proteins": n_prot,
       "classes": {k: {"terms": cls[k], "ia_weight": round(wt[k], 1)} for k in cls},
       "share_of_missed_weight": {k: round(wt[k] / tot_fn_w, 4) for k in wt if k.startswith("FN")},
       "top_branches_missed_ia": {b: round(w, 1) for b, w in rows},
       "top_branch_capture": {b: round(by_top_tp.get(b, 0.0) / (by_top_tp.get(b, 0.0) + w), 4)
                              for b, w in rows},
       "branch_names": {b: name_of.get(b, "?") for b, _ in rows},
       "examples": {k: v for k, v in examples.items()}}
json.dump(out, open(W / "curate_bp_errors.json", "w"), indent=1)
print(f"\n  wrote {W}/curate_bp_errors.json", flush=True)
print("DONE", flush=True)
