"""Phase 0: do pre-t0 abstracts NAME the novel BP branches we miss, leakage-audited?

THE QUESTION, and the gate for the whole abstract-signal build. The two cells we lose on the board
(LK-BP -0.072, PK-BP -0.076) are the prior-knowledge cells, where proteins have literature, and the
podium winners there win on text. Before building a per-protein abstract pipeline, prove the signal
exists AND survives leakage on a sample.

THE SIGNAL. For a sample of blind PK-BP and LK-BP proteins, take the true BP terms they are newly
annotated with (v227->v230) that our deployed system MISSES. Fetch each protein's UniProt-curated
literature, keep only references published BEFORE t0 (2025-09), fetch those abstracts, and measure:
does the pre-t0 abstract text NAME the missed term (its GO name or a synonym)? A high hit rate means
the function was in the literature before it was annotated, and we are not reading it.

THE LEAKAGE AUDIT, because a text signal is a mirage without it (memory: "honest-prior RED,
base-rate leak"). Two guards:
  (a) NOVELTY: exclude any gt term already annotated at t0. We query the t0 (v227) annotation state
      (read-only DB) per protein and drop terms already present, so we only test functions that were
      NOT yet annotated. A hit on those is genuinely predictive, not a transcription of an existing
      annotation.
  (b) SOURCE: UniProt's references are the curated papers, i.e. exactly the ones annotations may
      derive from. This probe REPORTS the residual: it flags that (b) needs the target-window GAF's
      per-annotation source PMID to fully close, and Phase 1 will add it. Phase 0's job is to see if
      the signal clears (a) at all.

WHAT A PASS MEANS: if a meaningful fraction of novel-missed terms are named in pre-t0 abstracts, the
signal is real and worth the pipeline. WHAT A FAIL MEANS: if abstracts rarely name the missed terms,
literature is not the lever here and we stop before building.

Read-only DB, external fetch (UniProt + NCBI, author-authorized), everything cached to disk so
re-runs cost nothing. No dispatch, no writes to the DB.
"""
import json, time, collections, urllib.request, urllib.parse, gzip, re
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, yaml, psycopg2

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
CACHE = W / "abstract_cache"; CACHE.mkdir(exist_ok=True)
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo")
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"
T0_YEAR = 2025   # t0 = 2025-09; keep references strictly before it (year < 2025, or 2025 pre-Sep)
T0_YM = (2025, 9)
N_PK, N_LK, SEED = 120, 60, 42
t0 = time.time()

# ---- ontology: names, synonyms, parents, alt_ids ----
par = collections.defaultdict(set); ns = {}; alt = {}; name = {}; syn = collections.defaultdict(set)
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("namespace: "):
        ns[cur_] = line[11:]
    elif cur_ and line.startswith("name: "):
        name[cur_] = line[6:]
    elif cur_ and line.startswith("synonym: "):
        m = re.search(r'"([^"]+)"', line)
        if m and (" EXACT " in line or " NARROW " in line):
            syn[cur_].add(m.group(1))
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

# match strings for a term: its name + exact/narrow synonyms, lowercased, length>=5 to avoid noise.
def match_strings(g):
    ss = {name.get(g, "")} | syn.get(g, set())
    return [s.lower() for s in ss if len(s) >= 5]

# the specific long content words of a term's name/synonyms (>=8 chars): these are the biology words
# (apoptotic, phosphorylation, differentiation) an abstract uses even when it paraphrases the GO name.
# Exact-phrase match is a hard FLOOR; long-word match is a looser proxy for what a semantic encoder,
# built in Phase 1, would actually capture. The truth sits between the two.
GENERIC = {"process", "regulation", "positive", "negative", "activity", "involved", "response",
           "pathway", "cellular", "biological", "function", "complex", "signaling"}
def long_words(g):
    out = set()
    for s in {name.get(g, "")} | syn.get(g, set()):
        for w in re.findall(r"[a-z]{8,}", s.lower()):
            if w not in GENERIC:
                out.add(w)
    return out

# ---- gt for PK and LK (aspect P), propagated ----
def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P":
                G[f[0]].add(alt.get(f[1], f[1]))
    return G
gt_pk, gt_lk = load_gt("groundtruth_PK.tsv"), load_gt("groundtruth_LK.tsv")

# ---- what the deployed system captures (its pool / submitted terms) ----
t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
c = np.asarray(t.column("category").to_pylist()); a = np.asarray(t.column("aspect").to_pylist())
def pool_of(cell):
    m = (c == cell) & (a == "bpo")
    P = np.asarray(t.column("protein_accession").to_pylist())[m]
    G = np.array([alt.get(g, g) for g in np.asarray(t.column("go_term_id").to_pylist())[m]])
    d = collections.defaultdict(set)
    for p_, g_ in zip(P, G):
        d[p_].add(g_)
    return d
pool_pk, pool_lk = pool_of("pk"), pool_of("lk")

rng = np.random.default_rng(SEED)
def sample(gt, pool, n):
    prots = sorted(set(gt) & set(pool))
    return sorted(rng.choice(prots, min(n, len(prots)), replace=False).tolist())
samp = [(p, "pk", gt_pk, pool_pk) for p in sample(gt_pk, pool_pk, N_PK)] + \
       [(p, "lk", gt_lk, pool_lk) for p in sample(gt_lk, pool_lk, N_LK)]
accs = [p for p, *_ in samp]
print(f"sample: {len(samp)} proteins ({N_PK} pk + {N_LK} lk)  ({time.time()-t0:.0f}s)", flush=True)

# ---- t0 (v227) BP annotation state per sample protein, for the NOVELTY guard ----
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
find(cfg); url = url.replace("postgresql+psycopg://", "postgresql://")
t0state = collections.defaultdict(set)
conn = psycopg2.connect(url); conn.set_session(readonly=True); cur = conn.cursor()
for s in range(0, len(accs), 500):
    cur.execute("""select protein_accession, go_id from protein_go_annotation pga
                   join go_term gt on gt.id = pga.go_term_id
                   where pga.annotation_set_id = %s and pga.protein_accession = any(%s)""",
                (V227, accs[s:s + 500]))
    for acc, g in cur.fetchall():
        g = alt.get(g, g)
        if g in BP:
            t0state[acc] |= {g} | anc(g)
conn.close()
print(f"t0 (v227) BP state loaded for {len(t0state)} proteins  ({time.time()-t0:.0f}s)", flush=True)


def fetch(u, dest, is_json=True):
    if dest.exists():
        return dest.read_text()
    req = urllib.request.Request(u, headers={"User-Agent": "protea-thesis-feasibility/1.0 "
                                             "(mailto:frapercan1@alum.us.es)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = r.read().decode("utf-8", "replace")
            dest.write_text(txt); time.sleep(0.34)   # NCBI etiquette: <3 req/s
            return txt
        except Exception as e:
            if attempt == 3:
                print(f"  fetch failed {u[:60]}: {e}", flush=True); return None
            time.sleep(1.5)


# ---- per protein: UniProt refs -> pre-t0 PMIDs ----
prot_pmids = {}
for i, acc in enumerate(accs):
    txt = fetch(f"https://rest.uniprot.org/uniprotkb/{acc}.json", CACHE / f"uniprot_{acc}.json")
    pmids = []
    if txt:
        try:
            for r in json.loads(txt).get("references", []):
                cit = r.get("citation", {})
                pm = [x["id"] for x in cit.get("citationCrossReferences", [])
                      if x.get("database") == "PubMed"]
                yr = str(cit.get("publicationDate", ""))[:4]
                if pm and yr.isdigit() and int(yr) < T0_YEAR:   # strictly pre-t0 (before 2025)
                    pmids.append(pm[0])
        except Exception:
            pass
    prot_pmids[acc] = sorted(set(pmids))
    if (i + 1) % 40 == 0:
        print(f"  uniprot {i+1}/{len(accs)}  ({time.time()-t0:.0f}s)", flush=True)
all_pmids = sorted({p for v in prot_pmids.values() for p in v})
print(f"pre-t0 PMIDs: {len(all_pmids):,} unique over {sum(1 for v in prot_pmids.values() if v)} "
      f"proteins with literature  ({time.time()-t0:.0f}s)", flush=True)

# ---- fetch abstracts in batches (efetch accepts many ids) ----
abstracts = {}
BATCH = 150
for s in range(0, len(all_pmids), BATCH):
    ids = all_pmids[s:s + BATCH]
    dest = CACHE / f"pubmed_{ids[0]}_{len(ids)}.xml"
    u = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id="
         + ",".join(ids))
    xml = fetch(u, dest)
    if not xml:
        continue
    for art in re.split(r"<PubmedArticle>", xml)[1:]:
        pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
        ab = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
        ti = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S)
        if pm:
            txt = (ti.group(1) if ti else "") + " " + ab
            abstracts[pm.group(1)] = re.sub(r"<[^>]+>", " ", txt).lower()
    print(f"  pubmed {min(s+BATCH,len(all_pmids))}/{len(all_pmids)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"abstracts fetched: {len(abstracts):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- the measurement ----
rows = []
for acc, cell, gt, pool in samp:
    closure = set().union(*[{g} | anc(g) for g in gt[acc]]) if gt[acc] else set()
    gt_bp = {g for g in closure if g in BP}
    novel = {g for g in gt_bp if g not in t0state.get(acc, set())}   # leakage guard (a)
    missed = {g for g in novel if g not in pool.get(acc, set())}     # what we fail to predict
    text = " ".join(abstracts.get(p, "") for p in prot_pmids.get(acc, []))
    has_text = bool(text.strip())
    def named(terms):   # exact GO name / synonym phrase in the abstract (the FLOOR)
        return sum(any(msx in text for msx in match_strings(g)) for g in terms)
    def loose(terms):   # any specific long biology word of the term present (looser proxy)
        return sum(bool(long_words(g)) and any(w in text for w in long_words(g)) for g in terms)
    rows.append({"acc": acc, "cell": cell, "n_pmids": len(prot_pmids.get(acc, [])),
                 "has_text": has_text, "gt_bp": len(gt_bp), "novel": len(novel),
                 "missed": len(missed),
                 "novel_named": named(novel), "missed_named": named(missed),
                 "novel_loose": loose(novel), "missed_loose": loose(missed)})

def agg(rs):
    rs = [r for r in rs if r["has_text"]]
    nv = sum(r["novel"] for r in rs); nn = sum(r["novel_named"] for r in rs)
    ms = sum(r["missed"] for r in rs); mn = sum(r["missed_named"] for r in rs)
    nvl = sum(r["novel_loose"] for r in rs); msl = sum(r["missed_loose"] for r in rs)
    return {"proteins_with_text": len(rs), "novel_terms": nv, "missed_terms": ms,
            "novel_hit_rate_exact": round(nn / nv, 4) if nv else None,
            "novel_hit_rate_loose": round(nvl / nv, 4) if nv else None,
            "missed_hit_rate_exact": round(mn / ms, 4) if ms else None,
            "missed_hit_rate_loose": round(msl / ms, 4) if ms else None}
res = {"question": "do pre-t0 abstracts name the novel BP branches we miss?",
       "sample": {"pk": N_PK, "lk": N_LK}, "t0": "2025-09", "leakage_guard_a": "novel = gt minus t0 state",
       "leakage_guard_b": "UniProt refs are curated sources; per-annotation source PMID audit deferred "
                          "to Phase 1 (needs target-window GAF).",
       "overall": agg(rows), "pk": agg([r for r in rows if r["cell"] == "pk"]),
       "lk": agg([r for r in rows if r["cell"] == "lk"]),
       "coverage": {"with_literature": sum(1 for r in rows if r["has_text"]),
                    "of_total": len(rows)}}
json.dump({"summary": res, "per_protein": rows}, open(W / "phase0_abstracts.json", "w"), indent=1)
print(f"\n=== do pre-t0 abstracts name what we miss? ({time.time()-t0:.0f}s) ===", flush=True)
print(f"  coverage: {res['coverage']['with_literature']}/{res['coverage']['of_total']} proteins have "
      f"pre-t0 literature", flush=True)
for k in ("overall", "pk", "lk"):
    a_ = res[k]
    print(f"  {k:8s}: missed-term hit  exact {a_['missed_hit_rate_exact']} / loose "
          f"{a_['missed_hit_rate_loose']}  (n={a_['missed_terms']}) | "
          f"all-novel exact {a_['novel_hit_rate_exact']} / loose {a_['novel_hit_rate_loose']} "
          f"(n={a_['novel_terms']})", flush=True)
print("  exact = GO name/synonym verbatim (FLOOR); loose = a specific long biology word present "
      "(proxy for a semantic encoder).", flush=True)
print("  A meaningful rate = the signal is real and pre-t0; leakage guard (b) still to close in Phase 1.", flush=True)
print("DONE", flush=True)
