"""Fetch UniProt text (desc + function comments + dated references) for ALL BP
training+eval proteins across both champion datasets (baseline LK + percut PK).

Single temporal cut applied at SCORING time = publications <= 2025-09 (v227 t0).
The cache stores raw year/month per reference so the cut is re-derivable.
Consolidated cache: literature_infame/uniprot_cache.json (seeded from prior caches).
"""
import os, json, time, requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "uniprot_cache.json")
NEED = os.path.join(HERE, "need_fetch.json")


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
        refs.append({"title": cit.get("title", "") or "", "year": y, "month": m})
    return {"acc": acc, "desc": desc, "funcs": funcs, "refs": refs}


def fetch_batch(accs):
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    for attempt in range(6):
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
    need = json.load(open(NEED))
    todo = [a for a in need if a not in cache]
    print(f"need={len(need)} already={len(need)-len(todo)} todo={len(todo)}", flush=True)
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
        if (i // CH) % 20 == 0:
            json.dump(cache, open(CACHE, "w"))
        print(f"  {i + len(chunk)}/{len(todo)} (got {len(results)})", flush=True)
        time.sleep(0.25)
    json.dump(cache, open(CACHE, "w"))
    print("done, cache size", len(cache), flush=True)


if __name__ == "__main__":
    main()
