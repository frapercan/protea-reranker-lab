"""At-scale rerun of the representation ablation: apples-to-apples on 100k training.

The crown (crown_train.py) trained every arm on the 15,000-protein reference that
sits on disk and its L48 control reached mean9 f_micro_w 0.1425 against the served
champion's 0.2150: the head was starved 6.7x. The apples-to-apples control
(crown_control_apples.py) proved that when the FULL learned encoder trains on the
champion's declared 100,000 pool and is scored on the SAME 15k reference, it
reaches 0.2197: the gap was data starvation, not the representation.

This script closes the loop by rerunning the crown's ARM comparison at that scale.
Each arm's Linear(d->2048)+kWTA head trains on the 100,000-protein pool base in
scale_pool_emb/ (pool GO leaves in scale_pool_go.json, pool-fit z-score stats),
and is then applied to the crown's own 7,401 queries and 15,000 reference proteins
for scoring. Training pool, query set and kNN reference are three DISTINCT sets;
in the crown the training base and the kNN reference were the same 15k set.

Question at scale: does per-dim z-score of the best fixed layer (L10-std) beat the
served champion's last-layer base (L48) AND the champion's 0.2150, and does a
learned softmax mix over layers {10,19,48} separate from L10-std? The crown found,
starved, that z-scoring was the lever (L10 raw did not beat L48, L10-std did,
p=6.9e-28) and the mix did NOT rediscover L10 (near-uniform, mild L19 peak). This
tests whether those verdicts hold when the head is no longer starved.

Arms differ ONLY in the base the shared head sees. The head recipe and all scoring
(cosine top-30 kNN GO transfer into the 15k reference, cafaeval f_micro_w over the
9 NK/LK/PK x MFO/BPO/CCO cells, protein-centric IA-Fmax for CIs/Wilcoxon) are
reused verbatim from crown_train.py / the lab. float32 everywhere (mid layers peak
at |4.9e5|; float16 overflows silently). No DB access; reads only pinned artefacts.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cafaeval.evaluation import cafa_eval
from scipy.stats import wilcoxon

from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise, propagate
from protea_reranker_lab.encoder_ablation import l2n, topk_real, sample_pairs  # noqa: F401

# --------------------------------------------------------------------------- config
W = "/home/frapercan/Thesis2/storage/layer_ablation"
REL = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(REL, "groundtruth_terms_of_interest.txt")

# Only the three layers the scale extraction produced (scale_extract.py): 10 = the
# ablation's best fixed layer, 48 = the served base, 19 = the one the crown mix mildly
# preferred. The scale mix is therefore over these three, not the crown's six.
LAYERS = [10, 19, 48]
ARMS = os.environ.get("SCALE_ARMS", "L48,L10,L10-std,mix-learned").split(",")
SEEDS = [int(x) for x in os.environ.get("SCALE_SEEDS", "42,43,44").split(",")]

EPOCHS = int(os.environ.get("SCALE_EPOCHS", "150"))
TRAIN_PAIRS = 300_000
KNN = 30
DICT_DIM = 2048
TOP_K = 128
BS = 32768
LR = 1e-3
OBJECTIVE = "cosine-lin"

NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
A2NS = {"mfo": "F", "bpo": "P", "cco": "C"}
CATS = {"NK": (os.path.join(REL, "groundtruth_NK.tsv"), None),
        "LK": (os.path.join(REL, "groundtruth_LK.tsv"), None),
        "PK": (os.path.join(REL, "groundtruth_PK.tsv"), os.path.join(REL, "groundtruth_PK_known.tsv"))}
ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- load data
print(f"[{time.strftime('%H:%M:%S')}] loading data (dev={DEV})", flush=True)
# scoring sets (unchanged from the crown): 7,401 queries + 15,000 kNN reference
qmeta = json.load(open(os.path.join(W, "emb_ankh_base/meta.json")))
qaccs = qmeta["accs"]
rmeta = json.load(open(os.path.join(W, "ref_emb/meta.json")))
raccs = rmeta["accs"]
ref_go = json.load(open(os.path.join(W, "ref_go.json")))
ref_go_list = [ref_go.get(a, []) for a in raccs]        # raw leaves for the scoring vote
seqs = json.load(open(os.path.join(W, "seqs.json")))    # query acc -> sequence

# training pool (100,000): the base the head learns on, scored on NEITHER query nor reference
pmeta = json.load(open(os.path.join(W, "scale_pool_meta.json")))
paccs = pmeta["accs"]
pool_go = json.load(open(os.path.join(W, "scale_pool_go.json")))
pool_go_list = [pool_go.get(a, []) for a in paccs]
assert not (set(paccs) & set(qaccs)), "pool overlaps queries: training leakage"
assert not (set(paccs) & set(raccs)), "pool overlaps reference: training leakage"

Player = {l: np.load(os.path.join(W, f"scale_pool_emb/layer_{l}.npy")).astype(np.float32) for l in LAYERS}
Qlayers = {l: np.load(os.path.join(W, f"emb_ankh_base/layer_{l}.npy")).astype(np.float32) for l in LAYERS}
Rlayers = {l: np.load(os.path.join(W, f"ref_emb/layer_{l}.npy")).astype(np.float32) for l in LAYERS}
D = Player[LAYERS[0]].shape[1]
assert all(Player[l].shape == (len(paccs), D) for l in LAYERS), "pool layer shape mismatch"
assert all(Qlayers[l].shape == (len(qaccs), D) for l in LAYERS)
assert all(Rlayers[l].shape == (len(raccs), D) for l in LAYERS)
print(f"[{time.strftime('%H:%M:%S')}] pool {len(paccs):,} x {D} | query {len(qaccs):,} | ref {len(raccs):,}",
      flush=True)

# GO DAG + closures. Training target is built from the POOL closures; the scoring vote uses
# the reference leaves (ref_go_list) directly, exactly as the crown does.
dag = GoDag.from_obo(OBO)
pool_closures = [propagate(pool_go_list[i], dag) for i in range(len(paccs))]
n_empty = sum(1 for c in pool_closures if not c)
print(f"[{time.strftime('%H:%M:%S')}] pool closures built; empty={n_empty}/{len(pool_closures)}", flush=True)

try:
    ni = json.load(open("/home/frapercan/Thesis2/storage/text_scorer/neighbor_identity.json"))
    ident_bucket = ni.get("bucket", {})
except Exception:
    ident_bucket = {}
qlen = {a: len(seqs.get(a, "")) for a in qaccs}


# --------------------------------------------------------------------------- bases
# z-score stats are fit on the TRAINING base (the 100k pool), then applied to the pool
# (train), the queries and the reference (encode). In the crown the fit set was the 15k
# reference because that WAS the training base; here the training base is the pool.
def zscore_stats(l):
    mu = Player[l].mean(0)
    sd = Player[l].std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


ZSTATS = {l: zscore_stats(l) for l in LAYERS}


def zed(bank, l):
    return ((bank[l] - ZSTATS[l][0]) / ZSTATS[l][1]).astype(np.float32)


zP = {l: zed(Player, l) for l in LAYERS}
zQ = {l: zed(Qlayers, l) for l in LAYERS}
zR = {l: zed(Rlayers, l) for l in LAYERS}

# per-arm base for training (pool) and for encoding (query, reference)
baseP = {"L48": Player[48], "L10": Player[10], "L10-std": zP[10]}
baseQ = {"L48": Qlayers[48], "L10": Qlayers[10], "L10-std": zQ[10]}
baseR = {"L48": Rlayers[48], "L10": Rlayers[10], "L10-std": zR[10]}
# mix arm: stacked z-scored layers (N, |LAYERS|, D); softmax mix keeps layers scale-comparable
mixP = np.stack([zP[l] for l in LAYERS], axis=1).astype(np.float32)
mixQ = np.stack([zQ[l] for l in LAYERS], axis=1).astype(np.float32)
mixR = np.stack([zR[l] for l in LAYERS], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- training (reused recipe)
def train_arm(arm, seed, ti_t, tj_t, y_t, npairs):
    """Learn the head on the POOL base, then encode the query + 15k reference through it.

    Returns (reference codes, query codes, softmax weights or None, final loss). The pool
    is training data only; its codes are never scored.
    """
    torch.manual_seed(seed)
    is_mix = (arm == "mix-learned")
    if is_mix:
        XP = torch.tensor(mixP, device=DEV)
        w = nn.Parameter(torch.zeros(len(LAYERS), device=DEV))
        enc = nn.Linear(D, DICT_DIM).to(DEV)
        params = list(enc.parameters()) + [w]

        def train_base():
            alpha = torch.softmax(w, dim=0)
            return F.normalize(torch.einsum("l,nld->nd", alpha, XP), dim=1)
    else:
        BP = torch.tensor(baseP[arm], device=DEV)
        PtN = F.normalize(BP, dim=1)
        d = BP.shape[1]
        w = None
        enc = nn.Linear(d, DICT_DIM).to(DEV)
        params = enc.parameters()

    opt = torch.optim.Adam(params, lr=LR)
    last_loss = float("nan")
    for _ in range(EPOCHS):
        z = enc(train_base()) if is_mix else enc(PtN)
        opt.zero_grad()
        loss = torch.zeros((), device=DEV)
        for b in range(0, npairs, BS):
            sl = slice(b, b + BS)
            zi, zj = z[ti_t[sl]], z[tj_t[sl]]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = loss + ((cos - y_t[sl]) ** 2).sum() / npairs
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())

    # encode the SCORING sets (query + 15k reference) through the trained head
    with torch.no_grad():
        if is_mix:
            alpha = torch.softmax(w, dim=0)
            zRc = enc(F.normalize(torch.einsum("l,nld->nd", alpha, torch.tensor(mixR, device=DEV)), dim=1))
            zQc = enc(F.normalize(torch.einsum("l,nld->nd", alpha, torch.tensor(mixQ, device=DEV)), dim=1))
            softmax_w = alpha.detach().cpu().numpy().tolist()
        else:
            zRc = enc(F.normalize(torch.tensor(baseR[arm], device=DEV), dim=1))
            zQc = enc(F.normalize(torch.tensor(baseQ[arm], device=DEV), dim=1))
            softmax_w = None
    Rc = topk_real(zRc.cpu().numpy().astype(np.float32), TOP_K)
    Qc = topk_real(zQc.cpu().numpy().astype(np.float32), TOP_K)
    del zRc, zQc, enc
    torch.cuda.empty_cache()
    return Rc, Qc, softmax_w, last_loss


# --------------------------------------------------------------------------- scoring (copied from crown_train.py)
def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return X / n


def knn_predict(Qn, Rn, outdir):
    os.makedirs(outdir, exist_ok=True)
    w = open(os.path.join(outdir, "p.tsv"), "w")
    B = 500
    for i in range(0, len(qaccs), B):
        sims = Qn[i:i + B] @ Rn.T
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
    if os.environ.get("SCALE_SKIP_CAFA"):
        return {k: 0.1 for k in [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]}
    cells = {}
    for cat, (gt, known) in CATS.items():
        _, best = cafa_eval(OBO, outdir, gt, ia=IA, no_orphans=True, norm="cafa",
                            prop="fill", exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
        for _, row in best["f_micro_w"].reset_index().iterrows():
            a = NS2A.get(row["ns"])
            if a:
                cells[f"{cat.lower()}-{a}"] = round(float(row["f_micro_w"]), 5)
    return cells


IAW = {}
for ln in open(IA):
    p = ln.rstrip("\n").split("\t")
    if len(p) == 2:
        try:
            IAW[p[0]] = float(p[1])
        except ValueError:
            pass
TOI_SET = {ln.strip() for ln in open(TOI) if ln.strip()}


def load_gt_cells():
    pk_known = defaultdict(set)
    for ln in open(CATS["PK"][1]):
        pr = ln.rstrip("\n").split("\t")
        if len(pr) >= 2 and pr[0] != "EntryID":
            pk_known[pr[0]].add(pr[1])
    cells = {}
    for cat, (gtf, _) in CATS.items():
        leaves = defaultdict(lambda: defaultdict(set))
        for ln in open(gtf):
            pr = ln.rstrip("\n").split("\t")
            if len(pr) < 3 or pr[0] == "EntryID":
                continue
            acc, go, asp = pr[0], pr[1], pr[2]
            if cat == "PK" and go in pk_known.get(acc, ()):
                continue
            asp_code = {"F": "mfo", "P": "bpo", "C": "cco"}.get(asp)
            if asp_code:
                leaves[asp_code][acc].add(go)
        for asp_code, accmap in leaves.items():
            target = A2NS[asp_code]
            cell = {}
            for acc, gos in accmap.items():
                prop = set()
                for g in gos:
                    prop |= dag.ancestors(g)
                prop = {t for t in prop if dag.aspect.get(t) == target and t in TOI_SET and t not in ROOTS}
                if prop:
                    cell[acc] = prop
            cells[(cat.lower(), asp_code)] = cell
    return cells


GT_CELLS = load_gt_cells()


def propagate_preds(outdir):
    raw = defaultdict(dict)
    for ln in open(os.path.join(outdir, "p.tsv")):
        acc, go, sc = ln.rstrip("\n").split("\t")
        raw[acc][go] = float(sc)
    out = {}
    for acc, gos in raw.items():
        fill = {}
        for go, sc in gos.items():
            for anc in dag.ancestors(go):
                if fill.get(anc, -1.0) < sc:
                    fill[anc] = sc
        out[acc] = fill
    return out


def fmax_protein(pred_fill, gt_terms, target_ns):
    gt_ia = sum(IAW.get(t, 0.0) for t in gt_terms)
    if gt_ia <= 0:
        return None
    items = []
    for t, s in pred_fill.items():
        if dag.aspect.get(t) == target_ns and t in TOI_SET and t not in ROOTS:
            items.append((s, IAW.get(t, 0.0), t in gt_terms))
    items.sort(key=lambda x: -x[0])
    cum_pred, cum_tp, bestf = 0.0, 0.0, 0.0
    for s, iaw, ingt in items:
        cum_pred += iaw
        if ingt:
            cum_tp += iaw
        if cum_pred > 0 and cum_tp > 0:
            pr = cum_tp / cum_pred
            rc = cum_tp / gt_ia
            f = 2 * pr * rc / (pr + rc)
            if f > bestf:
                bestf = f
    return bestf


def per_protein_fmax(outdir):
    preds = propagate_preds(outdir)
    res = {}
    for (cat, asp), gtmap in GT_CELLS.items():
        target = A2NS[asp]
        d = {}
        for acc, gt_terms in gtmap.items():
            pf = preds.get(acc, {})
            v = fmax_protein(pf, gt_terms, target)
            if v is not None:
                d[acc] = v
        res[f"{cat}-{asp}"] = d
    return res


# --------------------------------------------------------------------------- run all arms x seeds
ORDER = [f"{c}-{a}" for c in ("nk", "lk", "pk") for a in ("mfo", "bpo", "cco")]
raw = {arm: {"seeds": {}, "softmax_weights": {}, "final_loss": {}} for arm in ARMS}
perprot = {arm: {} for arm in ARMS}

for seed in SEEDS:
    print(f"\n[{time.strftime('%H:%M:%S')}] ===== SEED {seed} =====", flush=True)
    rng = np.random.default_rng(seed)
    pairs = sample_pairs(len(paccs), TRAIN_PAIRS, rng)           # pairs among POOL proteins
    ic = information_content(pool_closures, dag)
    best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in pool_closures]
    y = np.asarray(lin_pairwise(pool_closures, ic, pairs, best_ic), dtype=np.float32)
    ti_t = torch.tensor([p[0] for p in pairs], device=DEV)
    tj_t = torch.tensor([p[1] for p in pairs], device=DEV)
    y_t = torch.tensor(y, device=DEV)
    npairs = len(pairs)
    print(f"  pairs={npairs} lin_target mean={y.mean():.4f} nonzero={(y>0).mean():.3f}", flush=True)

    for arm in ARMS:
        t0 = time.time()
        Rc, Qc, sw, loss = train_arm(arm, seed, ti_t, tj_t, y_t, npairs)
        d = tempfile.mkdtemp(prefix=f"scale_{arm}_{seed}_")
        knn_predict(l2(Qc), l2(Rc), d)
        cells = nine(d)
        pp = per_protein_fmax(d)
        shutil.rmtree(d, ignore_errors=True)
        raw[arm]["seeds"][seed] = cells
        raw[arm]["softmax_weights"][seed] = sw
        raw[arm]["final_loss"][seed] = loss
        perprot[arm][seed] = pp
        m = float(np.mean([cells.get(k, 0.0) for k in ORDER]))
        print(f"  [{arm:14s}] loss={loss:.4f} mean9={m:.4f} ({time.time()-t0:.0f}s) "
              f"sw={['%.3f' % x for x in sw] if sw else '-'}", flush=True)

# --------------------------------------------------------------------------- aggregate
cell_mean = {arm: {k: float(np.mean([raw[arm]["seeds"][s].get(k, 0.0) for s in SEEDS])) for k in ORDER}
             for arm in ARMS}
cell_std = {arm: {k: float(np.std([raw[arm]["seeds"][s].get(k, 0.0) for s in SEEDS])) for k in ORDER}
            for arm in ARMS}

champ = json.load(open(os.path.join(W, "knn_confirm_results.json")))["d8979601-learned"]
ctrl_delta = {k: cell_mean["L48"].get(k, 0.0) - champ.get(k, 0.0) for k in ORDER}
ctrl_mae = float(np.mean([abs(v) for v in ctrl_delta.values()]))
ctrl_mean9_L48 = float(np.mean([cell_mean["L48"][k] for k in ORDER]))
champ_mean9 = float(np.mean([champ.get(k, 0.0) for k in ORDER]))


def unit_vectors():
    per_arm = {}
    for arm in ARMS:
        acc_vals = defaultdict(list)
        for s in SEEDS:
            for cell, d in perprot[arm][s].items():
                for acc, v in d.items():
                    acc_vals[(cell, acc)].append(v)
        per_arm[arm] = {k: float(np.mean(v)) for k, v in acc_vals.items() if len(v) == len(SEEDS)}
    common = set.intersection(*[set(per_arm[a].keys()) for a in ARMS])
    return per_arm, sorted(common)


UNIT, COMMON = unit_vectors()
print(f"\n[{time.strftime('%H:%M:%S')}] common per-protein units across arms: {len(COMMON)}", flush=True)


def boot_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


pp_pooled = {}
for arm in ARMS:
    v = [UNIT[arm][k] for k in COMMON]
    pp_pooled[arm] = {"mean_fmax": float(np.mean(v)), "ci95": boot_ci(v), "n": len(v)}

pp_cell = {arm: {} for arm in ARMS}
for arm in ARMS:
    by_cell = defaultdict(list)
    for (cell, acc) in COMMON:
        by_cell[cell].append(UNIT[arm][(cell, acc)])
    for cell in ORDER:
        vv = by_cell.get(cell, [])
        pp_cell[arm][cell] = {"mean_fmax": float(np.mean(vv)) if vv else None,
                              "ci95": boot_ci(vv) if vv else [None, None], "n": len(vv)}

# --------------------------------------------------------------------------- Wilcoxon + Holm
COMPARISONS = [c for c in [
    ("L10", "L48"),               # best fixed layer vs champion base
    ("L10-std", "L48"),           # z-scored best layer vs champion base
    ("L10-std", "L10"),           # is z-scoring the lever (crown: yes, p=6.9e-28)
    ("mix-learned", "L48"),       # learned mix vs champion base
    ("mix-learned", "L10-std"),   # does learned multi-layer beat best single-layer-std
] if c[0] in ARMS and c[1] in ARMS]
stats = []
for a, b in COMPARISONS:
    da = np.array([UNIT[a][k] for k in COMMON])
    db = np.array([UNIT[b][k] for k in COMMON])
    diff = da - db
    nz = diff[diff != 0]
    if len(nz) > 0:
        wstat, pval = wilcoxon(da, db, zero_method="wilcox", alternative="two-sided")
    else:
        wstat, pval = float("nan"), 1.0
    stats.append({"a": a, "b": b, "median_delta": float(np.median(diff)),
                  "mean_delta": float(np.mean(diff)), "delta_ci95": boot_ci(diff),
                  "frac_a_gt_b": float((diff > 0).mean()), "n": len(diff),
                  "wilcoxon_stat": float(wstat), "p_raw": float(pval)})
order_idx = np.argsort([s["p_raw"] for s in stats])
mtot = len(stats)
holm = [None] * mtot
prev = 0.0
for rank, i in enumerate(order_idx):
    adj = min(1.0, (mtot - rank) * stats[i]["p_raw"])
    adj = max(adj, prev)
    prev = adj
    holm[i] = adj
for i, s in enumerate(stats):
    s["p_holm"] = float(holm[i])
    s["sig_holm_0.05"] = bool(holm[i] < 0.05)


# --------------------------------------------------------------------------- stratification
def strat_group(assign):
    out = {arm: defaultdict(list) for arm in ARMS}
    for arm in ARMS:
        for (cell, acc) in COMMON:
            g = assign(acc)
            if g is not None:
                out[arm][g].append(UNIT[arm][(cell, acc)])
    return {arm: {g: {"mean_fmax": float(np.mean(v)), "n": len(v)} for g, v in d.items()}
            for arm, d in out.items()}


def len_band(acc):
    L = qlen.get(acc, 0)
    if L <= 318:
        return "short<=318"
    if 970 <= L <= 1959:
        return "long_970_1959"
    return None


strat_length = strat_group(len_band)
strat_identity = strat_group(lambda a: ident_bucket.get(a)) if ident_bucket else None

# --------------------------------------------------------------------------- receipt
report = {
    "experiment": "scale_representation_ablation",
    "question": ("At 100k training (no starvation), does L10-std beat L48 AND the served champion "
                 "(0.2150), and does a learned softmax mix over layers {10,19,48} separate from "
                 "L10-std? Retests the crown's starved verdicts (z-score is the lever; mix does not "
                 "rediscover L10)."),
    "design": ("Head trains on the 100,000-protein pool base (scale_pool_emb/, GO in "
               "scale_pool_go.json, pool-fit z-score stats), then encodes the crown's own 7,401 "
               "queries and 15,000 reference proteins for scoring. Training pool, query set and kNN "
               "reference are three DISTINCT sets. Compare to crown_result.json (starved 15k train) "
               "and crown_control_apples.json (full encoder, 100k train -> 0.2197)."),
    "hyperparameters": {
        "arms": ARMS, "seeds": SEEDS, "epochs": EPOCHS, "train_pairs": TRAIN_PAIRS,
        "knn_scoring": KNN, "dict_dim": DICT_DIM, "top_k": TOP_K, "batch_size": BS, "lr": LR,
        "objective": OBJECTIVE, "layers": LAYERS, "train_pool_n": len(paccs),
        "reference_n": len(raccs), "query_n": len(qaccs), "dtype": "float32", "device": DEV,
        "base_preprocessing": {
            "L48": "raw ankh-base layer 48 (d8979601's served base), then L2 in-head",
            "L10": "raw ankh-base layer 10, then L2 in-head",
            "L10-std": "per-dim z-score of L10 (mu,sd fit on the 100k POOL), then L2 in-head",
            "mix-learned": "learnable softmax over per-dim-z-scored layers {10,19,48} (pool-fit) -> "
                           "weighted sum -> L2 -> Linear(768,2048) -> kWTA128",
        },
        "zscore_fit_set": "100k training pool (the crown fit on the 15k reference, which was its base)",
        "cafaeval_settings": {"obo": OBO, "ia": IA, "toi": TOI, "prop": "fill", "norm": "cafa",
                              "no_orphans": True, "th_step": 0.01, "pk_excludes": "groundtruth_PK_known",
                              "metric": "f_micro_w"},
        "recipe_source": "storage/layer_ablation/crown_train.py (train + scoring reused verbatim)",
    },
    "cell_mean_f_micro_w": cell_mean,
    "cell_std_f_micro_w": cell_std,
    "per_seed_f_micro_w": {arm: raw[arm]["seeds"] for arm in ARMS},
    "final_loss": {arm: raw[arm]["final_loss"] for arm in ARMS},
    "mean9_f_micro_w": {arm: float(np.mean([cell_mean[arm][k] for k in ORDER])) for arm in ARMS},
    "control_check": {
        "served_champion": "d8979601 (knn_confirm_results.json, deployed encoder codes)",
        "champion_cells": champ, "champion_mean9": champ_mean9,
        "L48_control_cells": cell_mean["L48"], "L48_control_mean9": ctrl_mean9_L48,
        "per_cell_delta_L48_minus_champion": ctrl_delta,
        "mean_abs_error_vs_champion": ctrl_mae,
    },
    "learned_softmax_weights_mix": ({str(s): raw["mix-learned"]["softmax_weights"][s] for s in SEEDS}
                                    if "mix-learned" in ARMS else None),
    "softmax_weight_layer_order": LAYERS,
    "per_protein_fmax_pooled": pp_pooled,
    "per_protein_fmax_by_cell": pp_cell,
    "per_protein_metric_note": ("protein-centric IA-weighted Fmax on identical prediction/gt files; "
                                "companion to the term-centric f_micro_w headline. Used for CIs + Wilcoxon."),
    "wilcoxon_holm": {"family": [f"{a} vs {b}" for a, b in COMPARISONS],
                      "correction": "Holm-Bonferroni across the comparisons above",
                      "unit": "per (cell,protein) mean-over-seeds protein-centric IA-Fmax",
                      "results": stats},
    "stratification": {"length": strat_length, "identity": strat_identity,
                       "length_bands": {"short": "<=318", "long": "970-1959"}},
}
outp = os.path.join(W, "scale_result.json")
json.dump(report, open(outp, "w"), indent=2, default=float)
np.savez_compressed(os.path.join(W, "scale_perprot.npz"),
                    **{f"{arm}": np.array([UNIT[arm][k] for k in COMMON]) for arm in ARMS},
                    units=np.array([f"{c}|{a}" for (c, a) in COMMON]))

# --------------------------------------------------------------------------- console summary
print("\n=== SCALE RESULT: 9-cell f_micro_w (mean over seeds) ===", flush=True)
print(f"{'cell':12s}" + "".join(f"{a:>15s}" for a in ARMS) + f"{'champ':>15s}")
for k in ORDER:
    print(f"{k:12s}" + "".join(f"{cell_mean[a][k]:>15.4f}" for a in ARMS) + f"{champ.get(k,0):>15.4f}")
print(f"{'MEAN9':12s}" + "".join(f"{np.mean([cell_mean[a][kk] for kk in ORDER]):>15.4f}" for a in ARMS)
      + f"{champ_mean9:>15.4f}")
print(f"\nCONTROL L48 vs served champion: MAE={ctrl_mae:.4f}  mean9 L48={ctrl_mean9_L48:.4f} "
      f"champ={champ_mean9:.4f}", flush=True)
print("\n=== Wilcoxon / Holm (protein-centric IA-Fmax) ===")
for s in stats:
    print(f"  {s['a']:16s} vs {s['b']:14s} mean_delta={s['mean_delta']:+.4f} "
          f"p_raw={s['p_raw']:.2e} p_holm={s['p_holm']:.2e} sig={s['sig_holm_0.05']}")
if "mix-learned" in ARMS:
    print("\n=== learned softmax weights (mix-learned), layer order", LAYERS, "===")
    for s in SEEDS:
        print(f"  seed {s}: {['%.3f' % x for x in raw['mix-learned']['softmax_weights'][s]]}")
print(f"\nreceipt -> {outp}", flush=True)
