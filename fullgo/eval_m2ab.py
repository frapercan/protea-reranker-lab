"""Board-faithful 9-cell eval of the anc2vec-vs-sparse M2 predictions.

Runs cafaeval per category (NK/LK/PK) against the CAFA_forever groundtruth on
each prediction TSV, extracts per-aspect f_micro_w -> 9 cells, and prints the
anc2vec-vs-sparse delta (LK-BPO highlighted, the wall cell).
"""
import os, json
from pathlib import Path
from cafaeval.evaluation import cafa_eval

W = "/home/frapercan/Thesis2/storage/struct_gate/m2ab"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None),
        "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}


def nine_cell(pred_tsv):
    import tempfile, shutil
    cells = {}
    for cat, (gt, known) in CATS.items():
        d = tempfile.mkdtemp()
        shutil.copy(pred_tsv, os.path.join(d, "p.tsv"))
        _, best = cafa_eval(OBO, d, gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                            exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        for _, r in best["f_micro_w"].reset_index().iterrows():
            a = NS2ASP.get(r["ns"])
            if a:
                cells[f"{cat.lower()}-{a}"] = round(float(r["f_micro_w"]), 5)
        shutil.rmtree(d, ignore_errors=True)
    return cells


BASES = [b for b in ("anc2vec", "sparse", "gotext") if os.path.exists(os.path.join(W, f"pred_{b}.tsv"))]
res = {}
for basis in BASES:
    res[basis] = nine_cell(os.path.join(W, f"pred_{basis}.tsv"))
    print(f"[{basis}] {res[basis]}", flush=True)

order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
print("\n=== M2 label basis head-to-head (9 cells) ===")
hdr = f"{'cell':10s}" + "".join(f"{b:>10s}" for b in BASES)
print(hdr)
for k in order:
    row = f"{k:10s}" + "".join(f"{res[b].get(k, 0):>10.4f}" for b in BASES)
    star = "  <== wall" if k in ("lk-bpo", "pk-bpo") else ""
    print(row + star)
means = {b: sum(res[b].values()) / 9 for b in BASES}
print(f"{'MEAN':10s}" + "".join(f"{means[b]:>10.4f}" for b in BASES))
print("\nvs anc2vec: " + ", ".join(f"{b} {means[b]-means['anc2vec']:+.4f}" for b in BASES if b != "anc2vec"))
json.dump(res, open(os.path.join(W, "eval_9cell.json"), "w"), indent=2)
