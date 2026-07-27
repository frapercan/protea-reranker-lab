"""Controlled hard-neg arm at scale: isolate the objective from the pool size.

scale_train.py showed the cosine-lin L48 arm at 100k reaches mean9 f_micro_w 0.1422,
essentially equal to the crown's cosine-lin L48 at 15k (0.1425): training pool size
15k -> 100k is null for that arm. The served champion (hard-neg objective, ankh L48,
~100k) reaches 0.2150. The missing piece is a hard-neg arm trained in THIS harness so
the only difference from the 0.1422 cosine-lin L48 is the objective, not scale, not the
scoring reference.

This trains exactly that: ``_train_encoder`` with ArmSpec(objective="hard-neg") on the
same 100k pool L48 embeddings (scale_pool_emb/layer_48.npy) already on disk, encodes the
crown's 7,401 queries + 15,000 reference, scores with the identical crown protocol into
the 15k reference, cafaeval over the nine cells. If it lands near the champion's 0.2150
and far above 0.1422, the hard-neg objective, not the training pool size, is the lever
that the crown_control_apples run (receipt now missing) was really measuring.

Reads only pinned artefacts + the extracted pool embeddings. No DB. Writes one receipt.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections import defaultdict

import numpy as np
from cafaeval.evaluation import cafa_eval

from protea_reranker_lab.sdr import GoDag, propagate
from protea_reranker_lab.encoder_ablation import ArmSpec, EncoderAblationSpec, _train_encoder

W = "/home/frapercan/Thesis2/storage/layer_ablation"
REL = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(REL, "groundtruth_terms_of_interest.txt")
SEEDS = [int(x) for x in os.environ.get("HN_SEEDS", "42,43,44").split(",")]
KNN = 30
NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (os.path.join(REL, "groundtruth_NK.tsv"), None),
        "LK": (os.path.join(REL, "groundtruth_LK.tsv"), None),
        "PK": (os.path.join(REL, "groundtruth_PK.tsv"), os.path.join(REL, "groundtruth_PK_known.tsv"))}
ORDER = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]

print(f"[{time.strftime('%H:%M:%S')}] loading", flush=True)
qaccs = json.load(open(os.path.join(W, "emb_ankh_base/meta.json")))["accs"]
raccs = json.load(open(os.path.join(W, "ref_emb/meta.json")))["accs"]
paccs = json.load(open(os.path.join(W, "scale_pool_meta.json")))["accs"]
ref_go = json.load(open(os.path.join(W, "ref_go.json")))
ref_go_list = [ref_go.get(a, []) for a in raccs]
pool_go = json.load(open(os.path.join(W, "scale_pool_go.json")))

P = np.load(os.path.join(W, "scale_pool_emb/layer_48.npy")).astype(np.float32)   # 100k pool base
Q = np.load(os.path.join(W, "emb_ankh_base/layer_48.npy")).astype(np.float32)    # 7401 queries
R = np.load(os.path.join(W, "ref_emb/layer_48.npy")).astype(np.float32)          # 15000 reference
assert P.shape[1] == Q.shape[1] == R.shape[1] == 768
QR = np.vstack([Q, R]).astype(np.float32)   # encode both; split after
nq = len(qaccs)

dag = GoDag.from_obo(OBO)
pool_closures = [propagate(pool_go.get(a, []), dag) for a in paccs]
print(f"[{time.strftime('%H:%M:%S')}] pool {P.shape} closures built", flush=True)


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return X / n


def knn_predict(Qn, Rn, outdir):
    os.makedirs(outdir, exist_ok=True)
    w = open(os.path.join(outdir, "p.tsv"), "w")
    for i in range(0, len(qaccs), 500):
        sims = Qn[i:i + 500] @ Rn.T
        for r, srow in enumerate(sims):
            nn_idx = np.argpartition(-srow, KNN)[:KNN]
            votes = defaultdict(float)
            for n in nn_idx:
                s = float(srow[n])
                if s <= 0:
                    continue
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
            if a:
                cells[f"{cat.lower()}-{a}"] = round(float(row["f_micro_w"]), 5)
    return cells


arm = ArmSpec(name="L48-hardneg", kind="learned", dict_dim=2048, top_k=128, objective="hard-neg")
per_seed = {}
for seed in SEEDS:
    t0 = time.time()
    spec = EncoderAblationSpec(ref_n=len(paccs), knn=KNN, epochs=150, train_pairs=300_000, seed=seed)
    Rc, QRc = _train_encoder(P, QR, pool_closures, dag, arm, spec)   # Rc=pool codes (unused), QRc=query+ref
    Qc, R15c = QRc[:nq], QRc[nq:]
    d = tempfile.mkdtemp(prefix=f"hn_{seed}_")
    knn_predict(l2(Qc), l2(R15c), d)
    cells = nine(d)
    per_seed[seed] = cells
    m = float(np.mean([cells.get(k, 0.0) for k in ORDER]))
    print(f"  [L48-hardneg seed {seed}] mean9={m:.4f} ({time.time()-t0:.0f}s)", flush=True)

cell_mean = {k: float(np.mean([per_seed[s].get(k, 0.0) for s in SEEDS])) for k in ORDER}
mean9 = float(np.mean([cell_mean[k] for k in ORDER]))
champ = json.load(open(os.path.join(W, "knn_confirm_results.json")))["d8979601-learned"]
champ9 = float(np.mean([champ.get(k, 0.0) for k in ORDER]))
scale = json.load(open(os.path.join(W, "scale_result.json")))
cosine_l48 = scale["mean9_f_micro_w"]["L48"]

report = {
    "experiment": "scale_hardneg_control",
    "question": ("Isolate objective from pool size: hard-neg L48 at 100k in the SAME harness as the "
                 "cosine-lin arms. Does it reach the champion 0.2150 (=> objective is the lever), "
                 "leaving cosine-lin L48 at 0.1422 (scale null: 0.1425@15k ~= 0.1422@100k)?"),
    "arm": "learned-k128-hardneg, ankh L48, 100k pool, scored into 15k reference (crown protocol)",
    "seeds": SEEDS, "epochs": 150, "train_pairs": 300_000, "knn": KNN,
    "cell_mean_f_micro_w": cell_mean, "per_seed": per_seed, "mean9_f_micro_w": round(mean9, 5),
    "anchors": {"cosine_lin_L48_100k": round(cosine_l48, 5), "champion_hardneg": round(champ9, 5),
                "crown_cosine_lin_L48_15k": 0.1425},
    "delta_hardneg_minus_cosine_lin": round(mean9 - cosine_l48, 5),
    "delta_to_champion": round(mean9 - champ9, 5),
}
json.dump(report, open(os.path.join(W, "scale_hardneg_result.json"), "w"), indent=2, default=float)
print(f"\n=== L48-hardneg mean9={mean9:.4f} | cosine-lin L48={cosine_l48:.4f} | champion={champ9:.4f} ===")
print(f"hard-neg minus cosine-lin (objective lever, fixed 100k) = {mean9-cosine_l48:+.4f}")
print(f"receipt -> {os.path.join(W, 'scale_hardneg_result.json')}", flush=True)
