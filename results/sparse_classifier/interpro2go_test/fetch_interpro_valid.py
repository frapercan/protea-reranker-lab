"""Fetch precomputed InterPro xrefs for the 7401 query accessions from UniProtKB.

Uses the UniProtKB search API with fields=accession,xref_interpro, querying in
accession batches. Writes valid_protein2ipr.json {acc: [IPRxxxxxx,...]}.
"""
import json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
accs = [l.strip() for l in open(os.path.join(HERE, "valid_accessions.txt")) if l.strip()]
print(f"{len(accs)} accessions")

BASE = "https://rest.uniprot.org/uniprotkb/search"
CHUNK = 100  # accessions per query (keeps URL well under limits)
out = {}
session = requests.Session()

def fetch_chunk(chunk):
    q = " OR ".join(f"accession:{a}" for a in chunk)
    params = {"query": q, "fields": "accession,xref_interpro",
              "format": "tsv", "size": 500}
    for attempt in range(5):
        try:
            r = session.get(BASE, params=params, timeout=120)
            if r.status_code == 200:
                return r.text
            print(f"  status {r.status_code} attempt {attempt}", file=sys.stderr)
        except Exception as e:
            print(f"  err {e} attempt {attempt}", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("chunk failed")

nchunks = (len(accs) + CHUNK - 1) // CHUNK
for i in range(0, len(accs), CHUNK):
    chunk = accs[i:i + CHUNK]
    txt = fetch_chunk(chunk)
    lines = txt.strip().split("\n")
    # header: Entry  InterPro
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            acc = parts[0] if parts else ""
            if acc:
                out.setdefault(acc, [])
            continue
        acc, ipr_field = parts[0], parts[1]
        iprs = [x.strip() for x in ipr_field.split(";") if x.strip().startswith("IPR")]
        out[acc] = sorted(set(iprs))
    if (i // CHUNK) % 10 == 0:
        print(f"chunk {i//CHUNK+1}/{nchunks}  collected {len(out)}")

# ensure every queried acc present (even if no InterPro / not found)
missing = [a for a in accs if a not in out]
for a in missing:
    out[a] = []

json.dump(out, open(os.path.join(HERE, "valid_protein2ipr.json"), "w"))
with_ipr = sum(1 for a in accs if out.get(a))
print(f"DONE: {len(out)} entries, {with_ipr} with >=1 InterPro, {len(accs)-with_ipr} without")
print(f"not returned by UniProt at all: {len(missing)}")
