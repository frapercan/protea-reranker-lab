"""Shared: build InterPro2GO graded GO predictions from a protein->IPR map."""
import json, os, re, collections, functools

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"


@functools.lru_cache(maxsize=1)
def _parents():
    return json.load(open(os.path.join(SC, "go_parents.json")))


@functools.lru_cache(maxsize=1)
def ns_map():
    return json.load(open(os.path.join(HERE, "go_namespace.json")))


def ancestors(go, parents):
    seen, stack = set(), [go]
    while stack:
        g = stack.pop()
        for p in parents.get(g, []):
            if p not in seen:
                seen.add(p); stack.append(p)
    return seen


@functools.lru_cache(maxsize=1)
def ipr2go_prop():
    """IPR -> sorted list of propagated, non-obsolete, namespaced GO ids."""
    parents = _parents()
    nm = ns_map()
    ipr2go = collections.defaultdict(set)
    pat = re.compile(r"^InterPro:(IPR\d+).*?; (GO:\d+)\s*$")
    for line in open(os.path.join(HERE, "interpro2go.txt")):
        m = pat.match(line.strip())
        if m:
            ipr2go[m.group(1)].add(m.group(2))
    out = {}
    for ipr, gos in ipr2go.items():
        full = set()
        for g in gos:
            full.add(g); full |= ancestors(g, parents)
        out[ipr] = sorted(g for g in full if g in nm)
    return out


def interpro_preds(protein2ipr):
    """protein2ipr: {acc:[IPR...]} -> {acc: {go: graded_score}} (all aspects)."""
    i2g = ipr2go_prop()
    out = {}
    for acc, iprs in protein2ipr.items():
        iprs = [i for i in iprs if i in i2g]
        if not iprs:
            continue
        support = collections.Counter()
        for ipr in iprs:
            for g in i2g[ipr]:
                support[g] += 1
        n = len(iprs)
        out[acc] = {g: c / n for g, c in support.items()}
    return out
