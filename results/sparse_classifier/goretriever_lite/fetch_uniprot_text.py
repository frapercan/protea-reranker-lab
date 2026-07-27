"""Stage 1/2: fetch UniProt text (function comments + references) for NK+LK
test proteins, temporally filtered to the t0 cut (publications <= 2025-09).

Writes a per-protein cache JSON to scratchpad and a coverage feasibility.json.
"""
import os
import sys
import json
import time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get(
    "UNIPROT_CACHE",
    "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad/uniprot_cache.json",
)
T0_YEAR, T0_MONTH = 2025, 9  # inclusive cut


def parse_date(s):
    """Return (year, month_or_None). Date strings: '2015', '2015-03', '2015-03-12'."""
    if not s:
        return None, None
    parts = str(s).split("-")
    try:
        y = int(parts[0])
    except ValueError:
        return None, None
    m = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    return y, m


def date_bucket(y, m):
    """precut | future | ambiguous (year==2025 no month) | unknown."""
    if y is None:
        return "unknown"
    if y < T0_YEAR:
        return "precut"
    if y > T0_YEAR:
        return "future"
    # y == 2025
    if m is None:
        return "ambiguous"
    return "precut" if m <= T0_MONTH else "future"


def extract(entry):
    acc = entry["primaryAccession"]
    desc = ""
    pd_ = entry.get("proteinDescription", {})
    rn = pd_.get("recommendedName") or {}
    if rn.get("fullName", {}).get("value"):
        desc = rn["fullName"]["value"]
    elif pd_.get("submissionNames"):
        desc = pd_["submissionNames"][0].get("fullName", {}).get("value", "")
    funcs = []
    for c in entry.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts", []):
                if t.get("value"):
                    funcs.append(t["value"])
    refs = []
    for r in entry.get("references", []):
        cit = r.get("citation", {})
        y, m = parse_date(cit.get("publicationDate"))
        pmid = None
        for x in cit.get("citationCrossReferences", []):
            if x.get("database") == "PubMed":
                pmid = x.get("id")
        refs.append({
            "title": cit.get("title", "") or "",
            "year": y, "month": m, "bucket": date_bucket(y, m), "pmid": pmid,
        })
    return {"acc": acc, "desc": desc, "funcs": funcs, "refs": refs}


def fetch_batch(accs):
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    params = {"accessions": ",".join(accs), "format": "json",
              "fields": "accession,protein_name,cc_function,lit_pubmed_id"}
    # full json (fields param ignored for json content but harmless); retry
    for attempt in range(5):
        try:
            r = requests.get(url, params={"accessions": ",".join(accs),
                                          "format": "json"}, timeout=60)
            if r.status_code == 200:
                return r.json().get("results", [])
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed batch starting {accs[0]}")


def main():
    sets = json.load(open(os.path.join(HERE, "accession_sets.json")))
    want = sorted(set(sets["nk"]["all"]) | set(sets["lk"]["all"]))
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    todo = [a for a in want if a not in cache]
    print(f"want={len(want)} cached={len(cache)} todo={len(todo)}", flush=True)
    CH = 100
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        results = fetch_batch(chunk)
        got = set()
        for e in results:
            ex = extract(e)
            cache[ex["acc"]] = ex
            got.add(ex["acc"])
        # secondary accessions: map any missing to whatever came back
        for a in chunk:
            if a not in cache:
                cache[a] = {"acc": a, "desc": "", "funcs": [], "refs": [],
                            "_missing": True}
        json.dump(cache, open(CACHE, "w"))
        print(f"  {i + len(chunk)}/{len(todo)} (batch got {len(results)})",
              flush=True)
        time.sleep(0.3)
    json.dump(cache, open(CACHE, "w"))
    print("done, cache size", len(cache))


if __name__ == "__main__":
    main()
