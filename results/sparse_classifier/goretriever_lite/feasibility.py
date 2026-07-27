"""Stage 1 feasibility: coverage of temporally-valid text per category x aspect."""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad/uniprot_cache.json"


def has_precut_text(rec, mode):
    """mode='precut': function comment + pre-cut ref titles only.
       mode='all': function comment + all ref titles (leaky upper bound)."""
    if rec.get("funcs"):
        return True
    for r in rec["refs"]:
        if not r["title"]:
            continue
        if mode == "all":
            return True
        if r["bucket"] == "precut":
            return True
    return False


def n_precut_refs(rec, mode):
    n = 0
    for r in rec["refs"]:
        if not r["title"]:
            continue
        if mode == "all" or r["bucket"] == "precut":
            n += 1
    return n


def main():
    sets = json.load(open(os.path.join(HERE, "accession_sets.json")))
    cache = json.load(open(CACHE))

    report = {"t0_cut": "2025-09 (publications <= this are pre-cut)",
              "n_proteins_fetched": len(cache), "categories": {}}
    # ref date bucket tally (NK+LK)
    buckets = {"precut": 0, "future": 0, "ambiguous": 0, "unknown": 0,
               "no_title": 0}
    for cat in ("nk", "lk"):
        for acc in sets[cat]["all"]:
            for r in cache[acc]["refs"]:
                if not r["title"]:
                    buckets["no_title"] += 1
                else:
                    buckets[r["bucket"]] += 1
    report["ref_date_buckets_nk_lk"] = buckets

    for cat in ("nk", "lk", "pk"):
        if cat not in sets:
            continue
        accs_all = sets[cat]["all"]
        accs_bpo = set(sets[cat]["bpo"])
        # pk not in cache (not fetched) -> skip detailed
        if any(a not in cache for a in accs_all[:5]):
            report["categories"][cat] = {"note": "not fetched in stage 1"}
            continue
        c = {}
        for scope, accs in (("all_proteins", accs_all),
                            ("bpo_proteins", sorted(accs_bpo))):
            n = len(accs)
            func_only = sum(1 for a in accs if cache[a]["funcs"])
            precut = sum(1 for a in accs if has_precut_text(cache[a], "precut"))
            alltext = sum(1 for a in accs if has_precut_text(cache[a], "all"))
            any_ref_precut = sum(1 for a in accs
                                 if n_precut_refs(cache[a], "precut") > 0)
            multi_precut = sum(1 for a in accs
                               if (1 if cache[a]["funcs"] else 0)
                               + n_precut_refs(cache[a], "precut") >= 2)
            missing = sum(1 for a in accs if cache[a].get("_missing"))
            c[scope] = {
                "n": n,
                "has_function_comment": func_only,
                "has_function_comment_pct": round(100 * func_only / n, 1),
                "has_precut_text_headline": precut,
                "has_precut_text_pct": round(100 * precut / n, 1),
                "has_text_all_literature_leaky": alltext,
                "has_text_all_pct": round(100 * alltext / n, 1),
                "has_any_precut_ref_title": any_ref_precut,
                "has_>=2_precut_text_units": multi_precut,
                "missing_entries": missing,
            }
        report["categories"][cat] = c

    out = os.path.join(HERE, "feasibility.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
