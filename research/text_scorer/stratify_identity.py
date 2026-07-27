"""Neighbor-identity stratification of the text confirm: does the TEXT delta
concentrate where homology fails (twilight+no-hit+remote)? Re-runs cafaeval on the
saved prediction TSVs restricted to HARD-homology vs EASY query subsets.
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
print(f"HARD (twilight+no-hit+remote) {len(HARD)} | EASY (high+mod) {len(EASY)}")


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
for cfg in ("d8979601", "esm1b-raw", "protst-text", "combined"):
    p = os.path.join(W, "preds", cfg, "p.tsv")
    if not os.path.exists(p): continue
    out[cfg] = {"hard": eval_subset(p, HARD), "easy": eval_subset(p, EASY)}
    print(f"[{cfg}] hard-mean={sum(out[cfg]['hard'].values())/max(1,len(out[cfg]['hard'])):.4f} "
          f"easy-mean={sum(out[cfg]['easy'].values())/max(1,len(out[cfg]['easy'])):.4f}", flush=True)

print("\n=== TEXT delta (protst-text - esm1b-raw), HARD vs EASY homology ===")
pt, er = out.get("protst-text", {}), out.get("esm1b-raw", {})
for stratum in ("hard", "easy"):
    print(f"--- {stratum} homology ---")
    for k in [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]:
        d = pt.get(stratum, {}).get(k, 0) - er.get(stratum, {}).get(k, 0)
        star = "  <== text helps where homology fails" if (stratum == "hard" and k.endswith("bpo") and d > 0.005) else ""
        print(f"  {k:9s} text_delta={d:+.4f}{star}")
json.dump(out, open(os.path.join(W, "stratify_identity_result.json"), "w"), indent=2)
