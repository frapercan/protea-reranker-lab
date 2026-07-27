"""Neighbor-identity stratification of the ProTrek confirm: where does protrek-text's
signal live (vs champion d8979601 and vs protst-text) on HARD vs EASY homology?
Re-cafaevals the saved preds_protrek TSVs on query subsets by MMseqs2 identity bucket.
"""
import os, json, tempfile, shutil
from pathlib import Path
from cafaeval.evaluation import cafa_eval

W = "/home/frapercan/Thesis2/storage/text_scorer"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None), "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}

nid = json.load(open(os.path.join(W, "neighbor_identity.json")))["bucket"]
HARD = {a for a, b in nid.items() if b in ("twilight", "no-hit", "remote")}
EASY = {a for a, b in nid.items() if b in ("high", "mod")}
print(f"HARD {len(HARD)} | EASY {len(EASY)}")


def eval_subset(pred_tsv, keep):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "p.tsv"), "w") as w:
        for line in open(pred_tsv):
            if line.split("\t", 1)[0] in keep:
                w.write(line)
    cells = {}
    for cat, (gt, known) in CATS.items():
        try:
            _, best = cafa_eval(OBO, d, gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                                exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
            for _, r in best["f_micro_w"].reset_index().iterrows():
                a = NS2A.get(r["ns"])
                if a: cells[f"{cat.lower()}-{a}"] = round(float(r["f_micro_w"]), 5)
        except Exception:
            pass
    shutil.rmtree(d, ignore_errors=True)
    return cells


out = {}
for cfg in ("d8979601", "protst-text", "protrek-text", "combined"):
    p = os.path.join(W, "preds_protrek", cfg, "p.tsv")
    if not os.path.exists(p): continue
    out[cfg] = {"hard": eval_subset(p, HARD), "easy": eval_subset(p, EASY)}
    print(f"[{cfg}] hard={sum(out[cfg]['hard'].values())/max(1,len(out[cfg]['hard'])):.4f} "
          f"easy={sum(out[cfg]['easy'].values())/max(1,len(out[cfg]['easy'])):.4f}", flush=True)

order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
print("\n=== ProTrek vs champion, HARD vs EASY homology (protrek-text - d8979601) ===")
for stratum in ("hard", "easy"):
    print(f"--- {stratum} ---")
    for k in order:
        d = out.get("protrek-text", {}).get(stratum, {}).get(k, 0) - out.get("d8979601", {}).get(stratum, {}).get(k, 0)
        tag = "  <== CCO gain" if (k.endswith("cco") and d > 0.005) else ("  BP" if k.endswith("bpo") else "")
        print(f"  {k:9s} {d:+.4f}{tag}")
json.dump(out, open(os.path.join(W, "stratify_protrek_result.json"), "w"), indent=2)
