"""Aspect-aware label basis: BP terms from gotext M2, MF/CC from anc2vec M2.

Merges the two prediction TSVs per aspect (P->gotext, F/C->anc2vec) and evaluates
board-faithful 9 cells vs pure anc2vec. Needs go_id->aspect (from OBO namespace).
"""
import os, tempfile, shutil, json
from pathlib import Path
from collections import defaultdict
from cafaeval.evaluation import cafa_eval

W = "/home/frapercan/Thesis2/storage/struct_gate/m2ab"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None), "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}

# go_id -> namespace from OBO
ns = {}; cur = None
for line in open(OBO):
    line = line.strip()
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif line.startswith("namespace:") and cur: ns[cur] = line.split(" ", 1)[1]
BP = {g for g, n in ns.items() if n == "biological_process"}
print(f"BP terms in OBO: {len(BP)}", flush=True)

# merged: BP rows from gotext, non-BP rows from anc2vec
merged = os.path.join(W, "pred_aspectaware.tsv")
with open(merged, "w") as w:
    for line in open(os.path.join(W, "pred_gotext.tsv")):
        if line.split("\t")[1] in BP: w.write(line)
    for line in open(os.path.join(W, "pred_anc2vec.tsv")):
        if line.split("\t")[1] not in BP: w.write(line)


def nine(pred):
    cells = {}
    for cat, (gt, known) in CATS.items():
        d = tempfile.mkdtemp(); shutil.copy(pred, os.path.join(d, "p.tsv"))
        _, best = cafa_eval(OBO, d, gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                            exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        for _, r in best["f_micro_w"].reset_index().iterrows():
            a = NS2ASP.get(r["ns"])
            if a: cells[f"{cat.lower()}-{a}"] = round(float(r["f_micro_w"]), 5)
        shutil.rmtree(d, ignore_errors=True)
    return cells


aw = nine(merged)
anc = json.load(open(os.path.join(W, "eval_9cell.json")))["anc2vec"]
order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
print(f"\n{'cell':10s}{'anc2vec':>10s}{'aspect-aware':>14s}{'delta':>10s}")
for k in order:
    star = "  <== wall" if k in ("lk-bpo", "pk-bpo") else ""
    print(f"{k:10s}{anc[k]:>10.4f}{aw[k]:>14.4f}{aw[k]-anc[k]:>+10.4f}{star}")
ma = sum(anc.values())/9; mw = sum(aw.values())/9
print(f"{'MEAN':10s}{ma:>10.4f}{mw:>14.4f}{mw-ma:>+10.4f}")
json.dump({"anc2vec": anc, "aspect_aware": aw}, open(os.path.join(W, "eval_aspectaware.json"), "w"), indent=2)
