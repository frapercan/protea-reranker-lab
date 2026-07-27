"""Fetch pre-t0 abstracts for ALL eval PK-BP + LK-BP proteins, so the semantic candidate generator
can run over the blind window (not just the 180-protein feasibility sample).

Reuses Phase 0's exact fetch + cache (abstract_cache/). Resumable: everything already on disk is
skipped. UniProt refs -> pre-t0 (year<2025) PMIDs -> PubMed abstracts. NCBI etiquette (<3 req/s).
No DB, no dispatch. Author-authorized external fetch, same kind as Phase 0.
"""
import json, time, urllib.request, re
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
CACHE = W / "abstract_cache"; CACHE.mkdir(exist_ok=True)
T0_YEAR = 2025
t0 = time.time()

t = pq.read_table(W / "rerank_out" / "eval_scores.parquet")
c = np.asarray(t.column("category").to_pylist()); a = np.asarray(t.column("aspect").to_pylist())
P = np.asarray(t.column("protein_accession").to_pylist())
m = ((c == "pk") | (c == "lk")) & (a == "bpo")
accs = sorted(set(P[m].tolist()))
print(f"eval PK+LK BP proteins to fetch: {len(accs):,}  ({time.time()-t0:.0f}s)", flush=True)


def fetch(u, dest):
    if dest.exists():
        return dest.read_text()
    req = urllib.request.Request(u, headers={"User-Agent": "protea-thesis-feasibility/1.0 "
                                             "(mailto:frapercan1@alum.us.es)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = r.read().decode("utf-8", "replace")
            dest.write_text(txt); time.sleep(0.34)
            return txt
        except Exception as e:
            if attempt == 3:
                print(f"  fetch failed {u[:70]}: {e}", flush=True); return None
            time.sleep(1.5)


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
                if pm and yr.isdigit() and int(yr) < T0_YEAR:
                    pmids.append(pm[0])
        except Exception:
            pass
    prot_pmids[acc] = sorted(set(pmids))
    if (i + 1) % 200 == 0:
        nlit = sum(1 for v in prot_pmids.values() if v)
        print(f"  uniprot {i+1}/{len(accs)}  ({nlit} with pre-t0 lit)  ({time.time()-t0:.0f}s)", flush=True)
json.dump(prot_pmids, open(W / "eval_prot_pmids.json", "w"))
all_pmids = sorted({p for v in prot_pmids.values() for p in v})
print(f"pre-t0 PMIDs: {len(all_pmids):,} unique over "
      f"{sum(1 for v in prot_pmids.values() if v)} proteins  ({time.time()-t0:.0f}s)", flush=True)

# ---- which PMIDs still need fetching? (abstracts cached in batched xml files) ----
have = set()
for xmlf in CACHE.glob("pubmed_*.xml"):
    for pm in re.findall(r"<PMID[^>]*>(\d+)</PMID>", xmlf.read_text()):
        have.add(pm)
todo = [p for p in all_pmids if p not in have]
print(f"abstracts: {len(have):,} cached, {len(todo):,} to fetch  ({time.time()-t0:.0f}s)", flush=True)

BATCH = 150
for s in range(0, len(todo), BATCH):
    ids = todo[s:s + BATCH]
    dest = CACHE / f"pubmed_eval_{ids[0]}_{len(ids)}.xml"
    u = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id="
         + ",".join(ids))
    fetch(u, dest)
    if (s // BATCH + 1) % 20 == 0:
        print(f"  pubmed {min(s+BATCH,len(todo))}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE  ({time.time()-t0:.0f}s)", flush=True)
