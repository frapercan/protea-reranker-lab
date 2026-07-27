"""THE nine-cell board, regenerated from the frozen benchmark mirror.

Why this exists: `apps/web/lib/book.ts` published a NINE_CELL grid captioned "the sealed
board, f_micro_w, frame v227 to v230, first in 7 of 9". Byte for byte, those nine values
are the `baseline` block of the LAB's LOFO ablation
(protea-reranker-lab/results/sparse_classifier/lofo_9cell/result.json), a DIFFERENT
experiment. Every cell was wrong, PK-BP by 55% (0.141 published vs 0.2181 real), and the
grid contradicted its own page: pillar 4 says "the 0.213 we deliver" for that same cell.
Its receipt pointed at `storage/lofo_9cell/result.json`, a path that does not exist.

The board is third-party scored and frozen. This reads it and nothing else, so any number
it emits is regenerable by anyone in one command. No model, no training, no DB.
"""
import csv, json
from pathlib import Path

REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
NS = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
OURS = "predictions_protea.tsv"

# WHO IS OURS. Never guess this from the filename: `predictions_percutgraft.tsv` is a
# PROTEA variant and carries no "protea" substring, so a filename filter silently counts
# our own graft as a rival and reports 2/9 instead of 7/9. This campaign made exactly that
# mistake once already. The method table is the authority.
NAMES = {}
for line in open(REL / "method_names.tsv"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3 and parts[0] != "filename":
        NAMES[parts[0]] = parts[2]
OUR_FAMILY = {f for f, group in NAMES.items() if group.upper().startswith("PROTEA")}
assert OURS in OUR_FAMILY and len(OUR_FAMILY) >= 3, OUR_FAMILY

board, field = {}, {}
for cat in ("NK", "LK", "PK"):
    f = REL / f"results_{cat}" / "evaluation_best_f_micro_w.tsv"
    rows = [r for r in csv.DictReader(open(f), delimiter="\t") if r["f_micro_w"]]
    board[cat], field[cat] = {}, {}
    for ns, short in NS.items():
        cell = sorted((r for r in rows if r["ns"] == ns),
                      key=lambda r: -float(r["f_micro_w"]))
        ours = next(r for r in cell if r["filename"] == OURS)
        # the field = everyone outside the PROTEA family, per the method table
        ext = [r for r in cell if r["filename"] not in OUR_FAMILY]
        board[cat][short] = round(float(ours["f_micro_w"]), 4)
        field[cat][short] = {
            "best_external": ext[0]["filename"].replace("_predictions.tsv", "").replace(".tsv", ""),
            "best_external_f": round(float(ext[0]["f_micro_w"]), 4),
            "we_win": float(ours["f_micro_w"]) > float(ext[0]["f_micro_w"]),
            "gap": round(float(ours["f_micro_w"]) - float(ext[0]["f_micro_w"]), 4),
        }

won = sum(1 for c in board for a in board[c] if field[c][a]["we_win"])
mean = round(sum(board[c][a] for c in board for a in board[c]) / 9, 4)
out = {"source": str(REL), "our_file": OURS, "our_family_excluded_from_the_field": sorted(OUR_FAMILY),
       "board": board, "vs_external_field": field,
       "cells_won_vs_external_field": won, "unweighted_mean_of_nine": mean}
Path("/home/frapercan/Thesis2/storage/regen_headline/board_nine_cell.json").write_text(
    json.dumps(out, indent=1))

print("=== the nine-cell board, from the frozen mirror ===")
print(f"{'':4} {'MF':>8} {'BP':>8} {'CC':>8}")
for c in ("NK", "LK", "PK"):
    print(f"{c:4} " + " ".join(f"{board[c][a]:>8.4f}" for a in ("MF", "BP", "CC")))
print(f"\ncells won vs the EXTERNAL field: {won}/9")
for c in ("NK", "LK", "PK"):
    for a in ("MF", "BP", "CC"):
        d = field[c][a]
        if not d["we_win"]:
            print(f"  LOST {c}-{a}: {board[c][a]} vs {d['best_external']} {d['best_external_f']} ({d['gap']:+.4f})")
print(f"\nunweighted mean of the nine: {mean}")
print("NOTE: the published headline is 0.4063. If this mean differs, the headline is a")
print("      different statistic (per-category mean, or a different aggregation), and the")
print("      thesis must say WHICH. Do not assume they are the same number.")
