"""Fetch UniProt text for PK TEST (227->230) BP proteins at the TEST t0 cut
(publications <= 2025-09), so the PK literature arm is not artificially capped by
the prior NK+LK-only fetch. Same bucketing as goretriever_lite/fetch_uniprot_text.py.
Writes a dedicated cache. Leakage-clean: bucket boundary at 2025-09.
"""
import os, json, time, requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "test_pk_uniprot_cache.json")
ACC = json.load(open(os.path.join(HERE, "test_pk_bp_accessions.json")))["pk"]
T0_YEAR, T0_MONTH = 2025, 9


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
        refs.append({"title": cit.get("title", "") or "", "year": y, "month": m,
                     "bucket": date_bucket(y, m)})
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
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [a for a in ACC if a not in cache]
    print(f"want={len(ACC)} cached={len(cache)} todo={len(todo)}", flush=True)
    CH = 100
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        for e in fetch_batch(chunk):
            ex = extract(e)
            cache[ex["acc"]] = ex
        for a in chunk:
            if a not in cache:
                cache[a] = {"acc": a, "desc": "", "funcs": [], "refs": [], "_missing": True}
        if (i // CH) % 10 == 0:
            json.dump(cache, open(CACHE, "w"))
        print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(0.2)
    json.dump(cache, open(CACHE, "w"))
    print("done, cache size", len(cache))


if __name__ == "__main__":
    main()
