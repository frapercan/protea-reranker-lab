"""Board-faithful kNN GO-transfer confirm: winner layer/norm vs last-layer baseline.

Query = 7401 LAFA targets; reference = 15k t0-annotated subset. For each config,
kNN (top-K cosine) from query into reference, transfer neighbours' GO with a
cosine-weighted vote, then cafaeval f_micro_w per category (board-faithful),
stratified by length. Configs: baseline L48-dense (what PROTEA uses) vs
L10-dense-std and L10-kWTA128-std (the proxy winners). std stats are fit on the
REFERENCE pool (non-transductive) and applied to query + reference.
"""
import os, json, tempfile, shutil, numpy as np
from pathlib import Path
from collections import defaultdict
from cafaeval.evaluation import cafa_eval

W = "/home/frapercan/Thesis2/storage/layer_ablation"
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None), "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}
K = 30

qmeta = json.load(open(os.path.join(W, "emb_ankh_base/meta.json"))); qaccs = qmeta["accs"]
rmeta = json.load(open(os.path.join(W, "ref_emb/meta.json"))); raccs = rmeta["accs"]
ref_go = json.load(open(os.path.join(W, "ref_go.json")))
ref_go_list = [ref_go.get(a, []) for a in raccs]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n

def load(layer, which):
    d = "emb_ankh_base" if which == "q" else "ref_emb"
    return np.load(os.path.join(W, d, f"layer_{layer}.npy")).astype(np.float32)

def kwta(X, k):
    Xs = X.copy()
    for r in range(Xs.shape[0]):
        row = Xs[r]
        if k < row.size:
            row[np.abs(row) < np.partition(np.abs(row), -k)[-k]] = 0.0
    return Xs

def _std(layer):
    Q, R = load(layer, "q"), load(layer, "r")
    mu = R.mean(0); sd = R.std(0) + 1e-6
    return (Q - mu) / sd, (R - mu) / sd

def build(config):
    if config == "concat-L10L48-std":
        q10, r10 = _std(10); q48, r48 = _std(48)
        return l2(np.hstack([q10, q48])), l2(np.hstack([r10, r48]))
    if config == "concat-L10L48-kwta128":
        q10, r10 = _std(10); q48, r48 = _std(48)
        return (l2(kwta(np.hstack([q10, q48]), 128)), l2(kwta(np.hstack([r10, r48]), 128)))
    if config == "d8979601-learned":
        return (l2(np.load(os.path.join(W, "query_d8979601.npy")).astype(np.float32)),
                l2(np.load(os.path.join(W, "ref_d8979601.npy")).astype(np.float32)))
    if config == "L48-dense":
        return l2(load(48, "q")), l2(load(48, "r"))
    Q, R = load(10, "q"), load(10, "r")
    mu = R.mean(0); sd = R.std(0) + 1e-6  # std fit on reference pool
    Q = (Q - mu) / sd; R = (R - mu) / sd
    if config == "L10-kwta128-std":
        Q, R = kwta(Q, 128), kwta(R, 128)
    return l2(Q), l2(R)

def knn_predict(Qn, Rn, outdir):
    os.makedirs(outdir, exist_ok=True)
    w = open(os.path.join(outdir, "p.tsv"), "w")
    B = 500
    for i in range(0, len(qaccs), B):
        sims = Qn[i:i + B] @ Rn.T  # (b, Nref)
        for r, srow in enumerate(sims):
            nn = np.argpartition(-srow, K)[:K]
            votes = defaultdict(float)
            for n in nn:
                s = float(srow[n])
                if s <= 0: continue
                for go in ref_go_list[n]:
                    votes[go] += s
            if votes:
                mx = max(votes.values())
                acc = qaccs[i + r]
                for go, v in votes.items():
                    w.write(f"{acc}\t{go}\t{v / mx:.6f}\n")
    w.close()

def nine(outdir):
    cells = {}
    for cat, (gt, known) in CATS.items():
        _, best = cafa_eval(OBO, outdir, gt, ia=IA, no_orphans=True, norm="cafa",
                            prop="fill", exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        for _, row in best["f_micro_w"].reset_index().iterrows():
            a = NS2A.get(row["ns"])
            if a: cells[f"{cat.lower()}-{a}"] = round(float(row["f_micro_w"]), 5)
    return cells

results = {}
for config in ("L48-dense", "L10-kwta128-std", "concat-L10L48-std", "concat-L10L48-kwta128", "d8979601-learned"):
    Qn, Rn = build(config)
    d = tempfile.mkdtemp()
    knn_predict(Qn, Rn, d)
    results[config] = nine(d)
    m = sum(results[config].values()) / 9
    print(f"[{config}] mean={m:.4f} {results[config]}", flush=True)
    shutil.rmtree(d, ignore_errors=True)

base = sum(results["L48-dense"].values()) / 9
print("\n=== kNN CONFIRM (f_micro_w, ankh L10-std vs L48-dense baseline) ===")
order = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
print(f"{'cell':10s}" + "".join(f"{c:>16s}" for c in results))
for k in order:
    print(f"{k:10s}" + "".join(f"{results[c].get(k, 0):>16.4f}" for c in results))
print(f"{'MEAN':10s}" + "".join(f"{sum(results[c].values()) / 9:>16.4f}" for c in results))
json.dump(results, open(os.path.join(W, "knn_confirm_results.json"), "w"), indent=2)
