"""CG.2 STEP 1 -- the GATE (literature-based generation).

Does a text->GO GENERATOR built from a protein's OWN pre-t0 literature add TRUE BP-tail
IA-mass the deployed sequence-only pool lacks, at a precision MEANINGFULLY above CG.1's ~1%
co-occurrence floor (ideally toward the classifier's 11.59%)?

GENERATION not rescoring: for each target protein we embed its pre-t0 abstracts with
S-PubMedBert, score every BP GO term by MAX cosine(abstract, GO-definition), rank BP terms,
propose the top-k that are NOT already in the pool for that protein, and measure the true-tail
IA-mass ADDED (SET MEMBERSHIP, never a score -> dodges the scale confound of the refuted
text-as-scorer test). Accounting is byte-for-byte the CG.1 gate: propagate gt + pool through the
obo, incremental new mass = ancestors(proposal) - covered, true = & gt_prop, false = - gt_prop.

Leakage: PMIDs restricted to publicationDate year < 2025 (strictly before the 2025-09-04 t0
cutoff), re-derived here from the frozen uniprot cache; count of dropped >=2025 refs reported.
Frozen data only. No DB, no dispatch, no network (all abstracts cached). No cafaeval (that is
Step 2, gated on this precision).
"""
import json, collections, time, re
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq

t0 = time.time()
ROOT = Path("/home/frapercan/Thesis2")
R = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier"
EVAL = R / "percut_rerank/eval.parquet"
GTDIR = R / "lafa_gt"
OBO = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_F = ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
CACHE = ROOT / "storage/cooc_experiment/abstract_cache"
OUT = ROOT / "storage/regen_headline"
TOPKS = [5, 10, 25, 50]
T0_YEAR = 2025          # keep only PMIDs with pub year < 2025 (< 2025-09-04 cutoff, conservative)


def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


# ---------- obo: BP ancestors (is_a + part_of), alt_id, namespace, name, def ----------
par = collections.defaultdict(set); ns = {}; alt = {}; name = {}; deftxt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur = None
    elif line.startswith("id: GO:"):
        cur = line[4:]
    elif cur and line.startswith("namespace: "):
        ns[cur] = line[11:]
    elif cur and line.startswith("name: "):
        name[cur] = line[6:]
    elif cur and line.startswith("def: "):
        m = re.search(r'"([^"]+)"', line)
        if m:
            deftxt[cur] = m.group(1)
    elif cur and line.startswith("is_a: GO:"):
        par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"):
        par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}
log(f"obo: {len(ns):,} terms, {len(BP):,} BP")

_anc = {}
def anc(t):
    t = alt.get(t, t)
    if t in _anc: return _anc[t]
    seen, stack = set(), [t]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        for p in par.get(x, ()): stack.append(p)
    r = frozenset(x for x in seen if x in BP)
    _anc[t] = r
    return r

IA = {}
for line in open(IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try: IA[p[0]] = float(p[1])
        except ValueError: pass
def iamass(terms): return float(sum(IA.get(t, 0.0) for t in terms))
log(f"IA: {len(IA):,} terms")

# retrieval universe = BP terms with a definition. text = "name. definition"
BP_DEF = sorted(t for t in BP if t in deftxt)
BP_IDX = {t: i for i, t in enumerate(BP_DEF)}
log(f"BP terms with def (generator universe): {len(BP_DEF):,}")

# ---------- pre-t0 PMIDs per protein, re-derived from frozen uniprot cache ----------
def pmids_for(acc):
    p = CACHE / f"uniprot_{acc}.json"
    kept, dropped = [], 0
    if p.exists():
        try:
            for r in json.loads(p.read_text()).get("references", []):
                cit = r.get("citation", {})
                pm = [x["id"] for x in cit.get("citationCrossReferences", [])
                      if x.get("database") == "PubMed"]
                yr = str(cit.get("publicationDate", ""))[:4]
                if pm:
                    if yr.isdigit() and int(yr) < T0_YEAR:
                        kept.append(pm[0])
                    else:
                        dropped += 1
        except Exception:
            pass
    return sorted(set(kept)), dropped

# ---------- cached abstracts (NO network) ----------
abstracts = {}
for xmlf in CACHE.glob("pubmed_*.xml"):
    xml = xmlf.read_text()
    for art in re.split(r"<PubmedArticle>", xml)[1:]:
        pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
        ab = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
        ti = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S)
        if pm:
            txt = (ti.group(1) if ti else "") + " " + ab
            abstracts[pm.group(1)] = re.sub(r"<[^>]+>", " ", txt).strip()
log(f"cached abstracts parsed: {len(abstracts):,}")

# ---------- pool (BP terms per protein) from eval.parquet ----------
def load_pool(target_prots):
    tb = pq.read_table(EVAL, columns=["protein_accession", "go_term_id"]).to_pandas()
    tb = tb[tb.protein_accession.isin(target_prots)]
    tb["go"] = tb.go_term_id.map(lambda g: alt.get(g, g))
    tb = tb[tb.go.isin(BP)]
    return tb.groupby("protein_accession").go.apply(set).to_dict()

# ---------- encode GO defs + all needed abstracts ONCE (GPU) ----------
import torch
from sentence_transformers import SentenceTransformer
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=dev)
log(f"encoder on {dev}")
DEFV = enc.encode([f"{name.get(g,'')}. {deftxt.get(g,'')}" for g in BP_DEF], batch_size=256,
                  normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
DEFV_t = torch.from_numpy(DEFV).to(dev)                       # (|BP_DEF|, d)
log(f"encoded {len(BP_DEF):,} GO defs -> {DEFV.shape}")


def run_cell(cell, cat, gt_fn):
    log(f"===== {cell} =====")
    gt = pd.read_csv(GTDIR / gt_fn, sep="\t")
    gt = gt[gt.aspect == "P"].copy()
    gt["term"] = gt.term.map(lambda g: alt.get(g, g))
    gt_leaf = gt.groupby("EntryID").term.apply(set).to_dict()
    targets = sorted(gt_leaf)
    pool = load_pool(set(targets))

    # per-protein pre-t0 PMIDs (re-derived) + abstract availability
    prot_pmids = {}; dropped_total = 0
    for p in targets:
        pm, dr = pmids_for(p); prot_pmids[p] = pm; dropped_total += dr
    with_lit = [p for p in targets if prot_pmids[p]]
    prot_abs = {p: [x for x in prot_pmids[p] if x in abstracts] for p in targets}
    with_abs = [p for p in targets if prot_abs[p]]
    log(f"  {cell}: {len(targets)} targets | pool present {len(set(targets)&set(pool))} | "
        f"with pre-t0 PMID {len(with_lit)} | with cached abstract {len(with_abs)} | "
        f">=2025 refs dropped {dropped_total}")

    # propagate gt + pool; ceiling accounting (identical to CG.1)
    gt_prop = {p: frozenset().union(*[anc(t) for t in ts]) if ts else frozenset()
               for p, ts in gt_leaf.items()}
    pool_prop = {p: (frozenset().union(*[anc(t) for t in pool[p]]) if pool.get(p) else frozenset())
                 for p in targets}
    tp_pool = sum(iamass(gt_prop[p] & pool_prop[p]) for p in targets)
    fn_total = sum(iamass(gt_prop[p] - pool_prop[p]) for p in targets)
    gt_total = sum(iamass(gt_prop[p]) for p in targets)
    # FN reachable only over proteins that actually have abstracts (fair generator ceiling)
    fn_withabs = sum(iamass(gt_prop[p] - pool_prop[p]) for p in with_abs)
    log(f"  gt IA {gt_total:.1f} | pool-captured {tp_pool:.1f} ({100*tp_pool/gt_total:.1f}%) | "
        f"FN {fn_total:.1f} ({100*fn_total/gt_total:.1f}%) | FN over with-abstract prot {fn_withabs:.1f}")

    # encode the abstracts this cell needs (reuse global cache; encode unique once)
    used = sorted({x for p in with_abs for x in prot_abs[p]})
    PM_IDX = {p: i for i, p in enumerate(used)}
    PMV = enc.encode([abstracts[p][:2000] for p in used], batch_size=128, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    PMV_t = torch.from_numpy(PMV).to(dev)
    log(f"  encoded {len(used):,} abstracts for {cell}")

    added_true = {k: 0.0 for k in TOPKS}; added_false = {k: 0.0 for k in TOPKS}
    n_prop = {k: 0 for k in TOPKS}
    KMAX = max(TOPKS)
    for p in with_abs:
        rows = [PM_IDX[x] for x in prot_abs[p]]
        sc = torch.max(DEFV_t @ PMV_t[rows].T, dim=1).values.cpu().numpy()   # (|BP_DEF|,) max cosine
        order = np.argsort(-sc)
        present = pool.get(p, set())                 # raw pool BP terms -> "already in pool" = skip
        covered = set(pool_prop[p])                  # propagated coverage grows with proposals
        gtp = gt_prop[p]
        picks = 0
        for j in order:
            g = BP_DEF[j]
            if g in present:
                continue
            picks += 1
            new = anc(g) - covered
            tm = iamass(new & gtp); fm = iamass(new - gtp)
            covered |= new
            for k in TOPKS:
                if picks <= k:
                    added_true[k] += tm; added_false[k] += fm; n_prop[k] += 1
            if picks >= KMAX:
                break

    cell_res = {
        "target_proteins": len(targets),
        "pool_present": len(set(targets) & set(pool)),
        "proteins_with_pret0_pmid": len(with_lit),
        "proteins_with_cached_abstract": len(with_abs),
        "abstract_coverage_frac": round(len(with_abs) / len(targets), 4),
        "ge2025_refs_dropped": dropped_total,
        "gt_true_IA_mass": round(gt_total, 2),
        "pool_captured_IA_mass": round(tp_pool, 2),
        "pool_captured_frac": round(tp_pool / gt_total, 4),
        "FN_IA_mass_pool_misses": round(fn_total, 2),
        "FN_frac_of_gt": round(fn_total / gt_total, 4),
        "FN_IA_mass_over_with_abstract_proteins": round(fn_withabs, 2),
        "per_k": {},
    }
    for k in TOPKS:
        at, af = added_true[k], added_false[k]
        cell_res["per_k"][f"top{k}"] = {
            "proposals": n_prop[k],
            "true_IA_mass_added": round(at, 3),
            "false_IA_mass_added": round(af, 3),
            "generation_precision_IA": round(at / (at + af), 4) if (at + af) > 0 else None,
            "frac_of_FN_recovered": round(at / fn_total, 4) if fn_total > 0 else None,
            "frac_of_FN_withabs_recovered": round(at / fn_withabs, 4) if fn_withabs > 0 else None,
        }
        r = cell_res["per_k"][f"top{k}"]
        log(f"    top{k:<2}: +true {at:8.2f}  +false {af:9.2f}  prec {r['generation_precision_IA']}"
            f"  FN-rec {r['frac_of_FN_recovered']}  FN(abs)-rec {r['frac_of_FN_withabs_recovered']}")
    del PMV_t
    if dev == "cuda":
        torch.cuda.empty_cache()
    return cell_res


results = {
    "encoder": "pritamdeka/S-PubMedBert-MS-MARCO",
    "generator_universe_BP_with_def": len(BP_DEF),
    "t0_year_cutoff": T0_YEAR,
    "bars": {"cooccurrence_CG1_top5": {"LK": 0.0161, "PK": 0.0103},
             "classifier_generator": 0.1159},
    "cells": {},
}
results["cells"]["LK-BPO"] = run_cell("LK-BPO", "lk", "groundtruth_LK.tsv")
results["cells"]["PK-BPO"] = run_cell("PK-BPO", "pk", "groundtruth_PK.tsv")

json.dump(results, open(OUT / "cg2_step1_gate.json", "w"), indent=2)
log("wrote cg2_step1_gate.json")
print(json.dumps(results, indent=2))
