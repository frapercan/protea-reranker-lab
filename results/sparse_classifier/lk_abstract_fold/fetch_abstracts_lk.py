"""Fetch PubMed abstracts (NCBI EUtils efetch) for the PRECUT pmids cited by the
LK-BP train (v160..v225) + test (v227-v230) proteins not already covered by the
phaseA abstracts.json. Temporal honesty: only refs with publication date
<= 2025-09 (the v227 t0 cut) are eligible. Polite/batched. Seeds from the phaseA
cache; output: abstracts.json {pmid: abstract}.
"""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASEA = HERE.parent / "phaseA_separability" / "abstracts.json"
NEED = HERE / "need_pmids.json"
OUT = HERE / "abstracts.json"
LOG = HERE / "fetch_abstracts.log"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH = 200
SLEEP = 0.4  # polite (<3 req/s without key)


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def fetch_batch(pmids):
    data = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "abstract", "retmode": "xml",
        "tool": "protea-lkabsfold", "email": "frapercan1@alum.us.es",
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
    abstracts = {}
    if OUT.exists():
        abstracts = json.load(open(OUT))
    elif PHASEA.exists():
        abstracts = json.load(open(PHASEA))  # seed from phaseA
    log(f"seed abstracts: {len(abstracts)}")
    need = json.load(open(NEED))
    todo = [p for p in need if p not in abstracts]
    log(f"need={len(need)} todo={len(todo)}")
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(4):
            try:
                got = fetch_batch(chunk)
                abstracts.update(got)
                log(f"  batch {i // BATCH}: requested {len(chunk)}, "
                    f"got {len(got)}; total {len(abstracts)}")
                break
            except Exception as e:
                log(f"  batch {i // BATCH} attempt {attempt} ERR {e}")
                time.sleep(2 + attempt * 2)
        if (i // BATCH) % 5 == 0:
            json.dump(abstracts, open(OUT, "w"))
        time.sleep(SLEEP)
    json.dump(abstracts, open(OUT, "w"))
    json.dump({"need_pmids": len(need), "abstracts_total": len(abstracts)},
              open(HERE / "abstracts_fetch_meta.json", "w"), indent=2)
    log(f"DONE: total abstracts {len(abstracts)}")


if __name__ == "__main__":
    main()
