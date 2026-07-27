"""Fetch UniProt text for VALIDATION (225->227) BP proteins, dated and bucketed
to the EARLIER t0 cut (publications <= 2025-03, the v225 t0 ~ 2025-03-08).

Mirror of goretriever_lite/fetch_uniprot_text.py but with T0=2025-03 and the
validation accession list. Writes a separate cache to avoid clobbering the test
(2025-09) cache. Leakage-clean: only refs with bucket=='precut' are used as text.
"""
import os, json, time, requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "valid_uniprot_cache.json")
ACC = json.load(open(os.path.join(HERE, "valid_bp_accessions.json")))
T0_YEAR, T0_MONTH = 2025, 3  # inclusive cut == v225 t0


def parse_date(s):
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
    if y is None:
        return "unknown"
    if y < T0_YEAR:
        return "precut"
    if y > T0_YEAR:
        return "future"
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
        refs.append({"title": cit.get("title", "") or "", "year": y, "month": m,
                     "bucket": date_bucket(y, m), "pmid": pmid})
    return {"acc": acc, "desc": desc, "funcs": funcs, "refs": refs}


def fetch_batch(accs):
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    for attempt in range(5):
        try:
            r = requests.get(url, params={"accessions": ",".join(accs),
                                          "format": "json"}, timeout=90)
            if r.status_code == 200:
                return r.json().get("results", [])
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed batch starting {accs[0]}")


def main():
    want = ACC["all"]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [a for a in want if a not in cache]
    print(f"want={len(want)} cached={len(cache)} todo={len(todo)}", flush=True)
    CH = 100
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        results = fetch_batch(chunk)
        for e in results:
            ex = extract(e)
            cache[ex["acc"]] = ex
        for a in chunk:
            if a not in cache:
                cache[a] = {"acc": a, "desc": "", "funcs": [], "refs": [],
                            "_missing": True}
        if (i // CH) % 10 == 0:
            json.dump(cache, open(CACHE, "w"))
        print(f"  {i + len(chunk)}/{len(todo)} (batch got {len(results)})", flush=True)
        time.sleep(0.2)
    json.dump(cache, open(CACHE, "w"))
    print("done, cache size", len(cache))


if __name__ == "__main__":
    main()
