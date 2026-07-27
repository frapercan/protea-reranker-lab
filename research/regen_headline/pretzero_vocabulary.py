"""Regenerate the "97 percent of the missed terms already exist in the pre-window vocabulary" claim.

WHY THIS EXISTS. The claim is load-bearing on all three surfaces: it is what turns the BP result
from "we lack evidence" into "we fail to order evidence we already hold". `results.rst` states it,
`insights.rst` states its complement ("Only 3 percent are genuinely novel"), and `book.ts` leans on
it. Its only receipt is a markdown line in `BP_WALL_CHARACTERIZATION.md` ("true PK-BP pairs whose
term exists in the pre-t0 BP vocabulary: 0.970") and **no script regenerates it**. The scripts that
receipt cites include `fuse_and_score.py`, which is the file that carried the `rankpct` artefact.
A number on a surface with no script is the one thing this campaign refuses to ship, and this one
has been on three surfaces for weeks.

THE DEFINITION, made explicit because the claim never had one. "The pre-window vocabulary" is
stated but never defined, and the answer depends entirely on which definition is meant. Two are
available from frozen board data, and both are reported:

  V_known_PK   the BP terms already annotated to the PK proteins themselves at t0
               (`groundtruth_PK_known.tsv`, the board's own record of what was known before the
               window opened). This is the strict reading: the term was already in use *on these
               proteins*.
  V_known_all  the BP terms known at t0 across NK, LK and PK proteins pooled. The looser reading:
               the term was already in use *somewhere* in the corpus at t0.

The quantity: of the new PK-BP ground-truth pairs that appeared during the window
(`groundtruth_PK.tsv` aspect P), what fraction carries a term that already existed in the
vocabulary? Reported per definition, unweighted over pairs and weighted by information accretion,
because f_micro_w weights by IA and an unweighted fraction can flatter a claim about a weighted
metric.

WHAT THIS CAN AND CANNOT SETTLE. If a definition lands at 0.970 it identifies which one the claim
meant and gives it a script. If none does, **the claim as published is not reproducible from frozen
data and must come off the surfaces** rather than be re-derived into agreement. I am not going to
search definitions until one matches: both are declared here, before the numbers.
"""
import collections, json
from pathlib import Path

REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
IA_F = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
OUT = Path("/home/frapercan/Thesis2/storage/regen_headline")

IA = {}
for line in IA_F.open():
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass


def read(fn, aspect="P"):
    """(protein, term) pairs of `fn` restricted to one aspect. Files are EntryID/term/aspect."""
    out = set()
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 3 and c[2] == aspect:
                out.add((c[0], c[1]))
    return out


new_pk = read("groundtruth_PK.tsv")
known_pk = read("groundtruth_PK_known.tsv")
known_all = known_pk | read("groundtruth_NK.tsv") | read("groundtruth_LK.tsv")
# NK/LK "groundtruth_*" are the NEW annotations for those proteins, not their t0 knowledge; the
# only t0-knowledge file the board ships is the PK one. Say so rather than pretend otherwise.
V_known_PK = {t for _, t in known_pk}
V_known_all = {t for _, t in known_all}

res = {"claim": "97 percent of the true terms missed on PK-BP already exist in the pre-window "
                "vocabulary (results.rst; insights.rst states the 3 percent complement; book.ts leans on it)",
       "published_receipt": "BP_WALL_CHARACTERIZATION.md:55 -> 0.970, with NO script",
       "definitions_declared_before_the_numbers": {
           "V_known_PK": "BP terms already annotated to the PK proteins at t0 (groundtruth_PK_known.tsv)",
           "V_known_all": "V_known_PK plus the BP terms in the NK/LK groundtruth files. NOTE: those "
                          "files hold the NEW window annotations for those proteins, not their t0 "
                          "knowledge; the board ships a *_known file only for PK. This is therefore "
                          "an upper bound on the loose reading, not the loose reading itself."},
       "new_pk_bp_pairs": len(new_pk),
       "V_known_PK_terms": len(V_known_PK), "V_known_all_terms": len(V_known_all)}

for name, V in (("V_known_PK", V_known_PK), ("V_known_all", V_known_all)):
    hit = [(p, t) for (p, t) in new_pk if t in V]
    w_hit = sum(IA.get(t, 0.0) for _, t in hit)
    w_all = sum(IA.get(t, 0.0) for _, t in new_pk)
    res[name] = {
        "pairs_whose_term_is_already_in_the_vocabulary": len(hit),
        "unweighted_fraction": round(len(hit) / len(new_pk), 4),
        "ia_weighted_fraction": round(w_hit / w_all, 4) if w_all else None,
        "matches_the_published_0.970_unweighted": abs(len(hit) / len(new_pk) - 0.970) < 0.005,
    }
    print(f"  {name:12s} vocabulary {len(V):>6,} terms | "
          f"{len(hit):>6,}/{len(new_pk):,} pairs = {len(hit)/len(new_pk):.4f} unweighted, "
          f"{w_hit/w_all:.4f} IA-weighted", flush=True)

json.dump(res, open(OUT / "pretzero_vocabulary.json", "w"), indent=1)
print(f"\n  published claim: 0.970")
for name in ("V_known_PK", "V_known_all"):
    print(f"  {name:12s} unweighted {res[name]['unweighted_fraction']} "
          f"-> matches 0.970: {res[name]['matches_the_published_0.970_unweighted']}", flush=True)
print("\n  If neither matches, the published 0.970 is not reproducible from frozen board data and", flush=True)
print("  comes off the surfaces. I am not searching definitions until one agrees.", flush=True)
print("DONE", flush=True)
