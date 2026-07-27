"""Crown experiment of the representation ablation.

Question: the served learned k-WTA retrieval encoder ``d8979601`` sits on ankh-base's
LAST layer (L48), which the layer ablation flags as the WORST fixed base, and a learned
head has never been trained on any other base. Naive equal-weight concat(L10,L48) is worse
than L10 alone, so multi-layer helps only if the combination is LEARNED.

Five arms, SAME loss (MSE between cos(z_i,z_j) and IC-weighted Lin GO-semantic similarity of
the two proteins' propagated GO closures), SAME output dim 2048, SAME k=128, differing only in
the BASE the Linear(d->2048)+kWTA head sees:
  1. L48            control  -- d8979601's base; bounds training noise vs the served champion.
  2. L10            best fixed layer the ablation found.
  3. L10-std        per-dim z-score (stats fit on the REFERENCE pool only), then L2.
  4. mix-learned    learnable softmax over the 6 layers -> weighted sum -> Linear+kWTA.
  5. concat-learned concat the 6 layers (4608-d) -> Linear(4608,2048)+kWTA.

The head recipe (Linear(d,2048), topk_real k=128, Adam lr=1e-3, MSE on cos vs Lin) is REUSED
verbatim from ``protea_reranker_lab.encoder_ablation._train_encoder`` (helpers imported, not
reimplemented). Scoring copies the board-faithful protocol from ``knn_confirm.py`` (cosine
top-30 kNN GO transfer into the 15k reference, cosine-weighted vote, cafaeval f_micro_w over the
9 NK/LK/PK x MFO/BPO/CCO cells with the sealed settings).

float32 everywhere (layer 38 peaks at |490,564|; float16 overflows silently). No DB access.
Writes only under storage/layer_ablation/.
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

# ---- REUSED recipe helpers (imported, NOT reimplemented) ---------------------
from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise, propagate
from protea_reranker_lab.encoder_ablation import l2n, topk_real, sample_pairs

# --------------------------------------------------------------------------- config
W = "/home/frapercan/Thesis2/storage/layer_ablation"
REL = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(REL, "groundtruth_terms_of_interest.txt")

LAYERS = [0, 10, 19, 29, 38, 48]
ARMS = os.environ.get("CROWN_ARMS", "L48,L10,L10-std,mix-learned,concat-learned").split(",")
SEEDS = [int(x) for x in os.environ.get("CROWN_SEEDS", "42,43,44").split(",")]

# hyperparameters. Values _train_encoder exposes are matched to its dataclass defaults;
# values it does not fix (seed list) use the SAME setting for every arm (recorded here).
EPOCHS = int(os.environ.get("CROWN_EPOCHS", "150"))   # EncoderAblationSpec.epochs default
TRAIN_PAIRS = 300_000   # EncoderAblationSpec.train_pairs default
KNN = 30                # scoring top-K (knn_confirm) AND EncoderAblationSpec.knn default
DICT_DIM = 2048         # ArmSpec.dict_dim default
TOP_K = 128             # ArmSpec.top_k default
BS = 32768              # _train_encoder inner pair batch size
LR = 1e-3               # _train_encoder Adam lr
OBJECTIVE = "cosine-lin"  # all arms; hard-neg NOT used (arms differ only in base)

NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
A2NS = {"mfo": "F", "bpo": "P", "cco": "C"}
CATS = {"NK": (os.path.join(REL, "groundtruth_NK.tsv"), None),
        "LK": (os.path.join(REL, "groundtruth_LK.tsv"), None),
        "PK": (os.path.join(REL, "groundtruth_PK.tsv"), os.path.join(REL, "groundtruth_PK_known.tsv"))}
ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- load data
print(f"[{time.strftime('%H:%M:%S')}] loading data (dev={DEV})", flush=True)
qmeta = json.load(open(os.path.join(W, "emb_ankh_base/meta.json")))
qaccs = qmeta["accs"]                                   # 7401, query order
rmeta = json.load(open(os.path.join(W, "ref_emb/meta.json")))
raccs = rmeta["accs"]                                   # 15000, reference order
ref_go = json.load(open(os.path.join(W, "ref_go.json")))
ref_go_list = [ref_go.get(a, []) for a in raccs]        # raw leaves for the scoring vote
seqs = json.load(open(os.path.join(W, "seqs.json")))    # query acc -> sequence

Qlayers = {l: np.load(os.path.join(W, f"emb_ankh_base/layer_{l}.npy")).astype(np.float32) for l in LAYERS}
Rlayers = {l: np.load(os.path.join(W, f"ref_emb/layer_{l}.npy")).astype(np.float32) for l in LAYERS}
assert all(Qlayers[l].shape == (len(qaccs), 768) for l in LAYERS)
assert all(Rlayers[l].shape == (len(raccs), 768) for l in LAYERS)
for l in LAYERS:
    assert Qlayers[l].dtype == np.float32 and Rlayers[l].dtype == np.float32

# GO DAG + reference closures for the Lin training target
dag = GoDag.from_obo(OBO)
closures = [propagate(ref_go_list[i], dag) for i in range(len(raccs))]   # 15000 propagated sets
n_empty = sum(1 for c in closures if not c)
print(f"[{time.strftime('%H:%M:%S')}] closures built; empty={n_empty}/{len(closures)}", flush=True)

# neighbour identity buckets (query acc -> bucket), for stratification
try:
    ni = json.load(open("/home/frapercan/Thesis2/storage/text_scorer/neighbor_identity.json"))
    ident_bucket = ni.get("bucket", {})
except Exception:
    ident_bucket = {}

# query length band
qlen = {a: len(seqs.get(a, "")) for a in qaccs}


# --------------------------------------------------------------------------- bases
def zscore_stats(l):
    mu = Rlayers[l].mean(0)
    sd = Rlayers[l].std(0) + 1e-6            # fit on REFERENCE pool only
    return mu.astype(np.float32), sd.astype(np.float32)


ZSTATS = {l: zscore_stats(l) for l in LAYERS}
zR = {l: ((Rlayers[l] - ZSTATS[l][0]) / ZSTATS[l][1]).astype(np.float32) for l in LAYERS}
zQ = {l: ((Qlayers[l] - ZSTATS[l][0]) / ZSTATS[l][1]).astype(np.float32) for l in LAYERS}

baseR = {
    "L48": Rlayers[48], "L10": Rlayers[10], "L10-std": zR[10],
    "concat-learned": np.hstack([zR[l] for l in LAYERS]).astype(np.float32),   # 4608-d
}
baseQ = {
    "L48": Qlayers[48], "L10": Qlayers[10], "L10-std": zQ[10],
    "concat-learned": np.hstack([zQ[l] for l in LAYERS]).astype(np.float32),
}
# mix arm: stacked z-scored layers (N, 6, 768); softmax mix keeps layers scale-comparable
mixR = np.stack([zR[l] for l in LAYERS], axis=1).astype(np.float32)
mixQ = np.stack([zQ[l] for l in LAYERS], axis=1).astype(np.float32)

# NOTE recorded in receipt: mix-learned and concat-learned use per-layer z-score (ref-fit) so the
# learned combination is not dominated by raw scale (L38 |max| ~ 4.9e5 vs L10 ~ 5.9e4). L48/L10 use
# the RAW base to match d8979601's served recipe; L10-std uses z-score(L10) then L2 per the spec.

# --------------------------------------------------------------------------- training (reused recipe)
def train_arm(arm, seed, ti_t, tj_t, y_t, npairs):
    """Learn the GO-aligned Linear(d->2048)+topk head on ``arm``'s base. Returns ref/query codes."""
    torch.manual_seed(seed)
    is_mix = (arm == "mix-learned")
    if is_mix:
        XR = torch.tensor(mixR, device=DEV)
        w = nn.Parameter(torch.zeros(len(LAYERS), device=DEV))
        enc = nn.Linear(768, DICT_DIM).to(DEV)
        params = list(enc.parameters()) + [w]

        def base_ref():
            alpha = torch.softmax(w, dim=0)
            return F.normalize(torch.einsum("l,nld->nd", alpha, XR), dim=1)
    else:
        BR = torch.tensor(baseR[arm], device=DEV)
        RtN = F.normalize(BR, dim=1)                      # constant, == l2n(base)
        d = BR.shape[1]
        w = None
        enc = nn.Linear(d, DICT_DIM).to(DEV)
        params = enc.parameters()

    opt = torch.optim.Adam(params, lr=LR)
    last_loss = float("nan")
    for e in range(EPOCHS):
        z = enc(base_ref()) if is_mix else enc(RtN)
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
    # code extraction (topk_real reused)
    with torch.no_grad():
        if is_mix:
            alpha = torch.softmax(w, dim=0)
            zRc = enc(F.normalize(torch.einsum("l,nld->nd", alpha, XR), dim=1))
            XQ = torch.tensor(mixQ, device=DEV)
            zQc = enc(F.normalize(torch.einsum("l,nld->nd", alpha, XQ), dim=1))
            softmax_w = torch.softmax(w, dim=0).detach().cpu().numpy().tolist()
        else:
            zRc = enc(RtN)
            zQc = enc(F.normalize(torch.tensor(baseQ[arm], device=DEV), dim=1))
            softmax_w = None
    Rc = topk_real(zRc.cpu().numpy().astype(np.float32), TOP_K)
    Qc = topk_real(zQc.cpu().numpy().astype(np.float32), TOP_K)
    del zRc, zQc, enc
    torch.cuda.empty_cache()
    return Rc, Qc, softmax_w, last_loss


# --------------------------------------------------------------------------- scoring (copied from knn_confirm.py)
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
    if os.environ.get("CROWN_SKIP_CAFA"):
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


# --------------------------------------------------------------------------- per-protein IA-Fmax (companion metric)
# f_micro_w (headline) is a term-centric micro aggregate and is NOT per-protein decomposable.
# For CIs and the paired Wilcoxon test we compute, on the SAME prediction/gt files, the standard
# CAFA protein-centric IA-weighted Fmax. Our vote scores are already per-protein max-normalised
# (v/max), i.e. cafa-normalised, so Fmax is invariant to the tau grid. Root terms carry IA=0.
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
    """(cat,aspect) -> {acc: propagated gt term set restricted to aspect & TOI}. PK excludes PK_known."""
    pk_known = defaultdict(set)
    for ln in open(CATS["PK"][1]):
        pr = ln.rstrip("\n").split("\t")
        if len(pr) >= 2 and pr[0] != "EntryID":
            pk_known[pr[0]].add(pr[1])
    cells = {}
    for cat, (gtf, _) in CATS.items():
        leaves = defaultdict(lambda: defaultdict(set))   # aspect -> acc -> {go}
        for ln in open(gtf):
            pr = ln.rstrip("\n").split("\t")
            if len(pr) < 3 or pr[0] == "EntryID":
                continue
            acc, go, asp = pr[0], pr[1], pr[2]
            if cat == "PK" and go in pk_known.get(acc, ()):   # exclude known annotations
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
    """acc -> {term: max score} propagated (prop=fill), full ontology, from p.tsv."""
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
    """Continuous-threshold IA-weighted Fmax for one protein (aspect-restricted)."""
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
    res = {}   # cell -> {acc: fmax}
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
# perprot[arm][seed][cell][acc] = fmax
perprot = {arm: {} for arm in ARMS}

for seed in SEEDS:
    print(f"\n[{time.strftime('%H:%M:%S')}] ===== SEED {seed} =====", flush=True)
    rng = np.random.default_rng(seed)
    pairs = sample_pairs(len(raccs), TRAIN_PAIRS, rng)            # shared across arms this seed
    ic = information_content(closures, dag)
    best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]
    y = np.asarray(lin_pairwise(closures, ic, pairs, best_ic), dtype=np.float32)
    ti_t = torch.tensor([p[0] for p in pairs], device=DEV)
    tj_t = torch.tensor([p[1] for p in pairs], device=DEV)
    y_t = torch.tensor(y, device=DEV)
    npairs = len(pairs)
    print(f"  pairs={npairs} lin_target mean={y.mean():.4f} nonzero={(y>0).mean():.3f}", flush=True)

    for arm in ARMS:
        t0 = time.time()
        Rc, Qc, sw, loss = train_arm(arm, seed, ti_t, tj_t, y_t, npairs)
        d = tempfile.mkdtemp(prefix=f"crown_{arm}_{seed}_")
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

# --------------------------------------------------------------------------- aggregate: 9-cell table (mean over seeds)
cell_mean = {arm: {k: float(np.mean([raw[arm]["seeds"][s].get(k, 0.0) for s in SEEDS])) for k in ORDER}
             for arm in ARMS}
cell_std = {arm: {k: float(np.std([raw[arm]["seeds"][s].get(k, 0.0) for s in SEEDS])) for k in ORDER}
            for arm in ARMS}

# served champion reference (from knn_confirm on the deployed d8979601 codes)
champ = json.load(open(os.path.join(W, "knn_confirm_results.json")))["d8979601-learned"]
champ = {k.replace("nk-", "nk-").replace("lk-", "lk-").replace("pk-", "pk-"): v for k, v in champ.items()}
# control distance to champion
ctrl_delta = {k: cell_mean["L48"].get(k, 0.0) - champ.get(k, 0.0) for k in ORDER}
ctrl_mae = float(np.mean([abs(v) for v in ctrl_delta.values()]))
ctrl_mean9_L48 = float(np.mean([cell_mean["L48"][k] for k in ORDER]))
champ_mean9 = float(np.mean([champ.get(k, 0.0) for k in ORDER]))

# --------------------------------------------------------------------------- per-protein units (mean over seeds)
def unit_vectors():
    """arm -> {(cell,acc): mean fmax over seeds}. Units common to ALL arms (all seeds present)."""
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


# per-arm pooled protein-centric Fmax mean + bootstrap CI over proteins
pp_pooled = {}
for arm in ARMS:
    v = [UNIT[arm][k] for k in COMMON]
    pp_pooled[arm] = {"mean_fmax": float(np.mean(v)), "ci95": boot_ci(v), "n": len(v)}

# per-cell protein-centric mean (mean over seeds -> mean over proteins)
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
COMPARISONS = [
    ("L10", "L48"),               # best fixed layer vs champion base
    ("L10-std", "L48"),           # z-scored best layer vs champion base
    ("mix-learned", "L48"),       # learned softmax mix vs champion base (control)
    ("concat-learned", "L48"),    # learned concat vs champion base
    ("mix-learned", "L10-std"),   # does learned multi-layer beat best single-layer-std
    ("concat-learned", "mix-learned"),  # learned concat vs learned softmax mix
]
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
# Holm-Bonferroni across the 6 comparisons
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
    """assign: acc -> group label. Returns arm -> group -> mean fmax (over cell-protein units)."""
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
    "experiment": "crown_representation_ablation",
    "date": "2026-07-10",
    "question": ("Does a LEARNED multi-layer combination beat the served champion's last-layer base "
                 "(L48) and the best fixed layer (L10)? Where does the learned softmax put its mass?"),
    "hyperparameters": {
        "arms": ARMS, "seeds": SEEDS, "epochs": EPOCHS, "train_pairs": TRAIN_PAIRS,
        "knn_scoring": KNN, "dict_dim": DICT_DIM, "top_k": TOP_K, "batch_size": BS, "lr": LR,
        "objective": OBJECTIVE, "layers": LAYERS, "reference_n": len(raccs), "query_n": len(qaccs),
        "dtype": "float32", "device": DEV,
        "base_preprocessing": {
            "L48": "raw ankh-base layer 48 (d8979601's served base), then L2 in-head",
            "L10": "raw ankh-base layer 10, then L2 in-head",
            "L10-std": "per-dim z-score of L10 (mu,sd fit on REFERENCE pool only), then L2 in-head",
            "mix-learned": "learnable softmax over 6 per-dim-z-scored layers (ref-fit) -> weighted "
                           "sum -> L2 -> Linear(768,2048) -> kWTA128",
            "concat-learned": "concat 6 per-dim-z-scored layers (ref-fit, 4608-d) -> L2 -> "
                              "Linear(4608,2048) -> kWTA128",
        },
        "note_zscore_choice": ("mix/concat z-score each layer (ref-fit) so the learned combination is "
                               "not dominated by raw scale (L38 |max|~4.9e5 vs L10~5.9e4); recorded per "
                               "the instruction to fix unexposed choices identically across arms."),
        "cafaeval_settings": {"obo": OBO, "ia": IA, "toi": TOI, "prop": "fill", "norm": "cafa",
                              "no_orphans": True, "th_step": 0.01, "pk_excludes": "groundtruth_PK_known",
                              "metric": "f_micro_w"},
        "recipe_source": "protea_reranker_lab.encoder_ablation._train_encoder (helpers imported verbatim)",
        "scoring_source": "storage/layer_ablation/knn_confirm.py (protocol copied)",
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
    "learned_softmax_weights_mix": {str(s): raw["mix-learned"]["softmax_weights"][s] for s in SEEDS},
    "softmax_weight_layer_order": LAYERS,
    "per_protein_fmax_pooled": pp_pooled,
    "per_protein_fmax_by_cell": pp_cell,
    "per_protein_metric_note": ("protein-centric IA-weighted Fmax on identical prediction/gt files; "
                                "companion to the term-centric f_micro_w headline (which is not "
                                "per-protein decomposable). Used for CIs + Wilcoxon."),
    "wilcoxon_holm": {"family": [f"{a} vs {b}" for a, b in COMPARISONS],
                      "correction": "Holm-Bonferroni across the 6 comparisons above",
                      "unit": "per (cell,protein) mean-over-seeds protein-centric IA-Fmax",
                      "results": stats},
    "stratification": {"length": strat_length, "identity": strat_identity,
                       "length_bands": {"short": "<=318", "long": "970-1959"}},
}
outp = os.path.join(W, "crown_result.json")
json.dump(report, open(outp, "w"), indent=2, default=float)
np.savez_compressed(os.path.join(W, "crown_perprot.npz"),
                    **{f"{arm}": np.array([UNIT[arm][k] for k in COMMON]) for arm in ARMS},
                    units=np.array([f"{c}|{a}" for (c, a) in COMMON]))

# --------------------------------------------------------------------------- console summary
print("\n=== CROWN RESULT: 9-cell f_micro_w (mean over seeds) ===", flush=True)
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
print("\n=== learned softmax weights (mix-learned), layer order", LAYERS, "===")
for s in SEEDS:
    print(f"  seed {s}: {['%.3f' % x for x in raw['mix-learned']['softmax_weights'][s]]}")
print(f"\nreceipt -> {outp}", flush=True)
