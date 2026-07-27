"""Fetch PubMed abstracts (NCBI EUtils efetch) for the PRECUT pmids cited by the
hard PK/LK-BP proteins. Temporal honesty: only refs with publication date
<= 2025-09 (the v227 t0 cut) are eligible, reusing the bucket logic of the
literature_infame cache. Polite/batched. Output: abstracts.json {pmid: abstract}.
"""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "literature_infame" / "uniprot_cache.json"
HARD = HERE / "hard_frame.parquet"
OUT = HERE / "abstracts.json"
LOG = HERE / "fetch_abstracts.log"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
T0_YEAR, T0_MONTH = 2025, 9
BATCH = 200
SLEEP = 0.4  # polite (<3 req/s without key)


def is_precut(r):
    y, m = r.get("year"), r.get("month")
    if y is None:
        return True
    if y < T0_YEAR:
        return True
    if y > T0_YEAR:
        return False
    return (m is None) or (m <= T0_MONTH)


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def collect_pmids():
    cache = json.load(open(CACHE))
    hard = pd.read_parquet(HARD, columns=["protein_accession"])
    hp = set(hard.protein_accession.unique())
    pmids = set()
    prot_pmids = {}
    for p in hp:
        rec = cache.get(p) or {}
        pm = [str(r["pmid"]) for r in rec.get("refs", [])
              if r.get("pmid") and is_precut(r)]
        if pm:
            prot_pmids[p] = pm
        pmids.update(pm)
    return sorted(pmids), prot_pmids


def fetch_batch(pmids):
    data = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "abstract", "retmode": "xml",
        "tool": "protea-phaseA", "email": "frapercan1@alum.us.es",
    }).encode()
    req = urllib.request.Request(EUTILS, data=data)
    with urllib.request.urlopen(req, timeout=120) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    out = {}
    for art in root.iter("PubmedArticle"):
        pmid_el = art.find(".//PMID")
        if pmid_el is None:
            continue
        pmid = pmid_el.text
        parts = []
        title_el = art.find(".//ArticleTitle")
        if title_el is not None and title_el.text:
            parts.append(title_el.text)
        for ab in art.findall(".//Abstract/AbstractText"):
            txt = "".join(ab.itertext()).strip()
            if txt:
                label = ab.get("Label")
                parts.append(f"{label}: {txt}" if label else txt)
        if parts:
            out[pmid] = " ".join(parts)
    return out


def main():
    if LOG.exists():
        LOG.unlink()
    pmids, prot_pmids = collect_pmids()
    log(f"unique precut pmids to fetch: {len(pmids)}; "
        f"proteins with >=1 pmid: {len(prot_pmids)}")
    abstracts = {}
    if OUT.exists():
        abstracts = json.load(open(OUT))
        log(f"resuming, already have {len(abstracts)}")
    todo = [p for p in pmids if p not in abstracts]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(4):
            try:
                got = fetch_batch(chunk)
                abstracts.update(got)
                log(f"  batch {i // BATCH}: requested {len(chunk)}, "
                    f"got {len(got)} abstracts; total {len(abstracts)}")
                break
            except Exception as e:
                log(f"  batch {i // BATCH} attempt {attempt} ERR {e}")
                time.sleep(2 + attempt * 2)
        json.dump(abstracts, open(OUT, "w"))
        time.sleep(SLEEP)
    json.dump(abstracts, open(OUT, "w"))
    json.dump({"pmids_requested": len(pmids),
               "abstracts_fetched": len(abstracts),
               "proteins_with_pmid": len(prot_pmids)},
              open(HERE / "abstracts_fetch_meta.json", "w"), indent=2)
    log(f"DONE: {len(abstracts)}/{len(pmids)} abstracts")


if __name__ == "__main__":
    main()
