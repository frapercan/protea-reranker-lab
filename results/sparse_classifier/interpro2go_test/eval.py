"""Evaluate InterPro2GO BP signal on LAFA 227->230 (board-faithful cafa_eval).

(a) InterPro-ALONE bpo Fmax on LK + PK.
(b) Orthogonal recall: of true BP terms (propagated) our pool MISSES, what frac
    does InterPro recover? Plus overall true-BP recall comparison.
(c) UNION: add InterPro BP preds to our BP pool (max-score blend), re-score bpo.
All restricted to BP. PK excludes PK_known.
"""
import os, json, tempfile, shutil, collections
import pandas as pd
from cafaeval.evaluation import cafa_eval

HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
GT_DIR = os.path.join(SC, "lafa_gt")
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
GT = {
    "lk": (os.path.join(GT_DIR, "groundtruth_LK.tsv"), None,
           os.path.join(SC, "percut_rerank/baseline/predictions/lk/lk.tsv")),
    "pk": (os.path.join(GT_DIR, "groundtruth_PK.tsv"),
           os.path.join(GT_DIR, "groundtruth_PK_known.tsv"),
           os.path.join(SC, "percut_rerank/predictions/pk/pk.tsv")),
}

ns_map = json.load(open(os.path.join(HERE, "go_namespace.json")))
parents = json.load(open(os.path.join(SC, "go_parents.json")))

def ancestors(go):
    seen, stack = set(), [go]
    while stack:
        g = stack.pop()
        for p in parents.get(g, []):
            if p not in seen:
                seen.add(p); stack.append(p)
    return seen

# InterPro graded predictions -> dict[acc] -> dict[go]=score (BP only)
ip = pd.read_csv(os.path.join(HERE, "interpro_predictions.tsv"), sep="\t",
                 header=None, names=["acc", "go", "score"])
ip_bp = ip[ip["go"].map(lambda g: ns_map.get(g) == "bpo")]
ip_by_prot = collections.defaultdict(dict)
for r in ip_bp.itertuples(index=False):
    ip_by_prot[r.acc][r.go] = r.score
print(f"InterPro BP preds: {len(ip_bp)} rows, {len(ip_by_prot)} proteins")

def run_bpo(pred_dir, cat):
    gt_file, known, _ = GT[cat]
    _, dfs_best = cafa_eval(OBO, pred_dir, gt_file, ia=IA, no_orphans=True,
                            norm="cafa", prop="fill", exclude=known,
                            toi_file=TOI, th_step=0.01, n_cpu=8)
    best = dfs_best["f_micro_w"].reset_index()
    row = best[best["ns"] == "biological_process"]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return {"f_micro_w": float(row["f_micro_w"]), "tau": float(row["tau"]),
            "pr_micro_w": float(row.get("pr_micro_w", float("nan"))),
            "rc_micro_w": float(row.get("rc_micro_w", float("nan"))),
            "cov": float(row.get("cov", float("nan")))}

def write_dir(rows, cat):
    tmp = tempfile.mkdtemp(prefix=f"ip_{cat}_")
    df = pd.DataFrame(rows, columns=["acc", "go", "score"])
    df = df[df["score"] > 0.0]
    df.to_csv(os.path.join(tmp, f"{cat}.tsv"), sep="\t", header=False,
              index=False, float_format="%.6f")
    return tmp

result = {}
for cat in ("lk", "pk"):
    gt_file, known, pool_path = GT[cat]
    # ---- our BP pool ----
    pool = pd.read_csv(pool_path, sep="\t", header=None,
                       names=["acc", "go", "score"])
    pool_bp = pool[pool["go"].map(lambda g: ns_map.get(g) == "bpo")]
    pool_by_prot = collections.defaultdict(dict)
    for r in pool_bp.itertuples(index=False):
        pool_by_prot[r.acc][r.go] = max(pool_by_prot[r.acc].get(r.go, 0.0), r.score)

    # ---- true BP terms (propagated, TOI-filtered) per protein ----
    gt = pd.read_csv(gt_file, sep="\t")
    if known:
        kn = pd.read_csv(known, sep="\t")
        kn_pairs = set(zip(kn["EntryID"], kn["term"]))
    else:
        kn_pairs = set()
    toi = set(l.strip() for l in open(TOI) if l.strip())
    gt_bp = gt[gt["aspect"] == "P"]
    true_by_prot = collections.defaultdict(set)
    for r in gt_bp.itertuples(index=False):
        prop = {r.term} | ancestors(r.term)
        for t in prop:
            if ns_map.get(t) == "bpo" and t in toi:
                if known and (r.EntryID, t) in kn_pairs:
                    continue  # PK excludes known
                true_by_prot[r.EntryID].add(t)

    # ---- orthogonal recall analysis ----
    n_true = n_pool_hit = n_missed = n_ip_recover_missed = n_ip_hit_total = 0
    prot_with_recovery = 0
    for acc, trues in true_by_prot.items():
        pool_terms = set(pool_by_prot.get(acc, {}))
        ip_terms = set(ip_by_prot.get(acc, {}))
        for t in trues:
            n_true += 1
            inpool = t in pool_terms
            inip = t in ip_terms
            if inpool:
                n_pool_hit += 1
            if inip:
                n_ip_hit_total += 1
            if not inpool:
                n_missed += 1
                if inip:
                    n_ip_recover_missed += 1
        rec = {t for t in trues if t not in pool_terms and t in ip_terms}
        if rec:
            prot_with_recovery += 1

    ortho = {
        "n_true_bp_term_instances": n_true,
        "pool_recall": round(n_pool_hit / max(1, n_true), 4),
        "interpro_recall": round(n_ip_hit_total / max(1, n_true), 4),
        "n_true_missed_by_pool": n_missed,
        "n_missed_recovered_by_interpro": n_ip_recover_missed,
        "orthogonal_recall_of_missed": round(n_ip_recover_missed / max(1, n_missed), 4),
        "n_proteins_with_recovery": prot_with_recovery,
        "n_proteins_true_bp": len(true_by_prot),
    }

    # ---- (a) InterPro-alone bpo ----
    ip_rows = [(a, g, s) for a, d in ip_by_prot.items() for g, s in d.items()]
    d_alone = write_dir(ip_rows, cat)
    alone = run_bpo(d_alone, cat); shutil.rmtree(d_alone, ignore_errors=True)

    # ---- (c) UNION: max-score blend of pool BP + interpro BP ----
    union = collections.defaultdict(dict)
    for acc, d in pool_by_prot.items():
        for g, s in d.items():
            union[acc][g] = max(union[acc].get(g, 0.0), s)
    for acc, d in ip_by_prot.items():
        for g, s in d.items():
            union[acc][g] = max(union[acc].get(g, 0.0), s)
    union_rows = [(a, g, s) for a, d in union.items() for g, s in d.items()]
    d_union = write_dir(union_rows, cat)
    union_cell = run_bpo(d_union, cat); shutil.rmtree(d_union, ignore_errors=True)

    result[cat] = {"interpro_alone": alone, "union_maxblend": union_cell,
                   "orthogonality": ortho}
    print(cat, "alone", alone["f_micro_w"] if alone else None,
          "union", union_cell["f_micro_w"] if union_cell else None,
          "ortho_recall_missed", ortho["orthogonal_recall_of_missed"])

json.dump(result, open(os.path.join(HERE, "eval_raw.json"), "w"), indent=2)
print("written eval_raw.json")
