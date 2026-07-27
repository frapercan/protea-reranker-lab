"""What statistic is 0.4063? Declare the candidates first, then look.

WHY. 0.4063 is the sealed headline: it is what the thesis, the interface and the Sphinx book all
call the result. No file in the tree holds a nine-cell board that averages to it. The unweighted
mean of the nine frozen cells is 0.4076, which is close enough to look like the same number and
far enough that it is not. A headline nobody can regenerate is the one thing this campaign refuses
to ship, and this one is the headline.

THE DISCIPLINE THIS RUN EARNED, applied here: **declare every candidate before computing any of
them**, and if none matches, say so rather than inventing a tenth until one does. The 97 percent
claim was settled this way an hour ago: two definitions declared, one reproduced 0.970 at 0.9702.

THE CANDIDATES. The board's frozen mirror carries `.bak_prereranked`, `.bak_pregraft` and
`.bak_preinterpro` copies of `evaluation_best_f_micro_w.tsv`, so **the board has been recomputed at
least three times** and the current table is not the only one it has ever held. That makes "an
earlier board state" a candidate on the same footing as "a different statistic".

  1  unweighted mean of the nine current cells                        (known: 0.40764)
  2  mean weighted by each cell's ground-truth protein count
  3  mean weighted by each cell's ground-truth pair count
  4  unweighted mean of `f_micro` (the unweighted metric) not `f_micro_w`
  5  unweighted mean of the nine cells in each `.bak_*` board state   <- the board's own history
  6  unweighted mean over the other release windows on disk           (Sep_2025_Nov_2025, etc.)

THE GATE, quantity named: |candidate - 0.4063| < 0.0005. Anything that matches identifies the
statistic and gives the headline a script. **If nothing matches, 0.4063 is not reproducible from
the frozen board and the honest move is to say so and let the author decide**, not to publish
0.4076 as though it had always been the number.

NOTE ON WHOSE FIGURE THIS IS. The cells are the board's own, cited from its frozen table, so any
mean of them is a statistic over board figures and stays in the board's frame. That is the one
family of absolute numbers we may quote (see WE_DO_NOT_REPRODUCE_THE_BOARD.md).
"""
import csv, json, collections, statistics
from pathlib import Path

ROOT = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases")
REL = ROOT / "Sep_2025_Mar_2026"
OURS = "predictions_protea.tsv"
NS = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
TARGET = 0.4063

res = {"target": TARGET, "candidates_declared_before_computing": [
    "unweighted mean of the nine current cells",
    "mean weighted by gt protein count",
    "mean weighted by gt pair count",
    "unweighted mean of f_micro (not f_micro_w)",
    "unweighted mean of each .bak_* board state (the board's own history)",
    "unweighted mean over the other release windows on disk"],
    "gate": "|candidate - 0.4063| < 0.0005", "results": {}}


def cells(rel: Path, fname: str, col: str = "f_micro_w"):
    """The nine cells of `fname` in release `rel`, or as many as exist."""
    out = {}
    for cat in ("NK", "LK", "PK"):
        f = rel / f"results_{cat}" / fname
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(), delimiter="\t"):
            if r.get("filename") == OURS and r.get("ns") in NS:
                try:
                    out[f"{cat}-{NS[r['ns']]}"] = float(r[col])
                except (ValueError, KeyError, TypeError):
                    pass
    return out


def record(name, vals, weights=None):
    if len(vals) != 9:
        res["results"][name] = {"n_cells": len(vals), "note": "incomplete, not nine cells"}
        print(f"  {name:52s} only {len(vals)} cells", flush=True)
        return
    if weights:
        m = sum(vals[k] * weights[k] for k in vals) / sum(weights[k] for k in vals)
    else:
        m = statistics.mean(vals.values())
    hit = abs(m - TARGET) < 0.0005
    res["results"][name] = {"mean": round(m, 5), "delta_vs_0.4063": round(m - TARGET, 5),
                            "MATCHES": hit, "cells": {k: round(v, 4) for k, v in vals.items()}}
    print(f"  {name:52s} {m:.5f}  delta {m-TARGET:+.5f}  -> matches: {hit}", flush=True)


# ground-truth sizes per cell, for the weighted candidates
prot, pair = collections.Counter(), collections.Counter()
for cat in ("NK", "LK", "PK"):
    seen = collections.defaultdict(set)
    with (REL / f"groundtruth_{cat}.tsv").open() as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 3:
                a = {"P": "BP", "F": "MF", "C": "CC"}.get(c[2])
                if a:
                    seen[f"{cat}-{a}"].add(c[0])
                    pair[f"{cat}-{a}"] += 1
    for k, v in seen.items():
        prot[k] = len(v)

cur = cells(REL, "evaluation_best_f_micro_w.tsv")
record("1 unweighted mean, current board", cur)
record("2 weighted by gt protein count", cur, prot)
record("3 weighted by gt pair count", cur, pair)
record("4 unweighted mean of f_micro (not _w)", cells(REL, "evaluation_best_f_micro.tsv", "f_micro")
       or cells(REL, "evaluation_best_f_micro_w.tsv", "f_micro"))

for bak in ("evaluation_best_f_micro_w.tsv.bak_prereranked",
            "evaluation_best_f_micro_w.tsv.bak_pregraft",
            "evaluation_best_f_micro_w.tsv.bak_preinterpro"):
    record(f"5 unweighted mean, {bak.split('.bak_')[1]}", cells(REL, bak))

for other in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "Sep_2025_Mar_2026"):
    record(f"6 unweighted mean, {other.name}", cells(other, "evaluation_best_f_micro_w.tsv"))

hits = [k for k, v in res["results"].items() if v.get("MATCHES")]
res["verdict"] = ("0.4063 is: " + hits[0]) if hits else (
    "NO declared candidate reproduces 0.4063. It is not derivable from the frozen board, and the "
    "author decides what the headline should be. Do not silently publish 0.4076 in its place.")
json.dump(res, open(Path(__file__).parent / "rederive_0_4063.json", "w"), indent=1)
print(f"\n=== target {TARGET} ===\n  {res['verdict']}", flush=True)
print("DONE", flush=True)
