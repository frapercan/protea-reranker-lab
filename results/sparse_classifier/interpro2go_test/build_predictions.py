"""Build InterPro2GO predictions for the 7401 query proteins.

- interpro2go.txt -> IPR -> [GO]
- propagate each IPR's GO set up the DAG (go_parents.json)
- per protein: term score = (#IPRs supporting term) / (#IPRs of protein)  [graded]
  also emit a constant-1.0 variant for robustness.
- parse OBO for GO -> namespace.
Writes:
  interpro_predictions.tsv          (graded, all aspects, 3-col)
  interpro_predictions_const.tsv    (constant 1.0)
  go_namespace.json                 (GO -> bpo/mfo/cco)
  ipr2go_prop.json, build_stats.json
"""
import json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
PARENTS = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/go_parents.json"

parents = json.load(open(PARENTS))

def ancestors(go):
    seen, stack = set(), [go]
    while stack:
        g = stack.pop()
        for p in parents.get(g, []):
            if p not in seen:
                seen.add(p); stack.append(p)
    return seen

# ---- parse OBO for namespace + obsolete ----
ns_map = {}
obsolete = set()
NS3 = {"biological_process": "bpo", "molecular_function": "mfo",
       "cellular_component": "cco"}
cur, cur_ns, cur_obs = None, None, False
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur, cur_ns, cur_obs = None, None, False
    elif line.startswith("id: GO:"):
        cur = line[4:].strip()
    elif line.startswith("namespace:") and cur:
        cur_ns = NS3.get(line.split(":", 1)[1].strip())
        if cur_ns:
            ns_map[cur] = cur_ns
    elif line.startswith("is_obsolete: true") and cur:
        obsolete.add(cur)
print(f"OBO namespaces: {len(ns_map)} terms, {len(obsolete)} obsolete")

# ---- parse interpro2go ----
ipr2go = collections.defaultdict(set)
pat = re.compile(r"^InterPro:(IPR\d+).*?; (GO:\d+)\s*$")
for line in open(os.path.join(HERE, "interpro2go.txt")):
    m = pat.match(line.strip())
    if m:
        ipr2go[m.group(1)].add(m.group(2))
print(f"interpro2go: {len(ipr2go)} IPRs with GO")

# propagate each IPR GO set
ipr2go_prop = {}
for ipr, gos in ipr2go.items():
    full = set()
    for g in gos:
        full.add(g)
        full |= ancestors(g)
    # keep only non-obsolete terms with known namespace
    full = {g for g in full if g in ns_map}
    ipr2go_prop[ipr] = sorted(full)

# ---- per protein predictions ----
prot2ipr = json.load(open(os.path.join(HERE, "protein2ipr.json")))
graded_rows = []
const_rows = []
bp_terms_per_prot = []
nterms_per_prot = []
mapped_ipr_prot = 0
for acc, iprs in prot2ipr.items():
    iprs_with_go = [i for i in iprs if i in ipr2go_prop]
    if not iprs_with_go:
        continue
    mapped_ipr_prot += 1
    support = collections.Counter()
    for ipr in iprs_with_go:
        for g in ipr2go_prop[ipr]:
            support[g] += 1
    n = len(iprs_with_go)
    nterms_per_prot.append(len(support))
    bp_terms_per_prot.append(sum(1 for g in support if ns_map.get(g) == "bpo"))
    for g, c in support.items():
        graded_rows.append((acc, g, c / n))
        const_rows.append((acc, g, 1.0))

with open(os.path.join(HERE, "interpro_predictions.tsv"), "w") as f:
    for a, g, s in graded_rows:
        f.write(f"{a}\t{g}\t{s:.6f}\n")
with open(os.path.join(HERE, "interpro_predictions_const.tsv"), "w") as f:
    for a, g, s in const_rows:
        f.write(f"{a}\t{g}\t{s:.6f}\n")
json.dump(ns_map, open(os.path.join(HERE, "go_namespace.json"), "w"))
json.dump({k: v for k, v in ipr2go_prop.items()},
          open(os.path.join(HERE, "ipr2go_prop.json"), "w"))

import statistics as st
stats = {
    "n_query": len(prot2ipr),
    "n_with_any_ipr": sum(1 for v in prot2ipr.values() if v),
    "n_with_go_mapped_ipr": mapped_ipr_prot,
    "mean_go_terms_per_prot": round(st.mean(nterms_per_prot), 2) if nterms_per_prot else 0,
    "mean_bp_terms_per_prot": round(st.mean(bp_terms_per_prot), 2) if bp_terms_per_prot else 0,
    "bp_fraction_of_terms": round(sum(bp_terms_per_prot) / max(1, sum(nterms_per_prot)), 4),
    "total_pred_rows": len(graded_rows),
}
json.dump(stats, open(os.path.join(HERE, "build_stats.json"), "w"), indent=2)
print(json.dumps(stats, indent=2))
