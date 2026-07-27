"""Phase 3b: the DECIDER. temporally-gated f_micro_w on LK-BPO / PK-BPO in the TRUE board frame.

Anchor A = the deployed reranker pool submission (eval_scores.parquet reranker_score), scored by
cafa_eval with lab obo+IA, prop=fill, norm=cafa, no_orphans, toi; PK adds -known (evaluation.nf:279).
Arm B = A + the method's GENERATED extras (top-K BP terms/protein NOT in the pool), scored by the
method's own overlap, submitting the better half. Delta (B-A) must clear the 0.0034 fold-noise floor.

Methods compared as generators (all projected through the IDENTICAL Phase-2 head recipe, isolating
the GO representation):
  LEARNED  = learned k-WTA GO codes (this build)
  TWOTOWER = fixed-recipe two-tower go_sparse_codes (unlearned SVD->kWTA)
  DENSE    = dense annotation-RAG aligner (armA scores)          [no k-WTA]
  FIXEDSCORE control = LEARNED candidates with a RANDOM score     [volume/scale artifact test]

cafa_eval runs under the PROTEA venv. Read-only w.r.t. repos; writes only under storage/.
"""
import json, collections, subprocess, tempfile, time, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
OUT = Path("/home/frapercan/Thesis2/storage/kwta_go_encoder")
CW = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
T0D = Path("/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025")
OBO = str(T0D / "go-basic.obo"); IA_F = str(T0D / "IA.tsv")
TOI = str(REL / "groundtruth_terms_of_interest.txt")
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
LEARNED = sys.argv[1] if len(sys.argv) > 1 else "coann_struct_text"
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- ontology ----
par = collections.defaultdict(set); ns = {}; alt = {}; cur = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]": cur = None
    elif line.startswith("id: GO:"): cur = line[4:]
    elif cur and line.startswith("namespace: "): ns[cur] = line[11:]
    elif cur and line.startswith("is_a: GO:"): par[cur].add(line[6:].split(" ! ")[0].strip())
    elif cur and line.startswith("relationship: part_of GO:"): par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur and line.startswith("alt_id: GO:"): alt[line[8:]] = cur
BP = {t for t, n in ns.items() if n == "biological_process"}

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",no_orphans=True,
    toi_file="{toi}",{exc}max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''

def cafa(rows, GT, known):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p_, g_, v in rows: fh.write(f"{p_}\t{g_}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p_, ts in GT.items():
                for g_ in ts: fh.write(f"{p_}\t{g_}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        exc = f'exclude="{known}",' if known else ""
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA_F, toi=TOI, o=str(raw), exc=exc))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            log(f"cafa FAIL: {r.stderr[-500:]}"); return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return round(float(best["f_micro_w"]), 5) if best else None

def load_gt(fn):
    G = collections.defaultdict(set)
    with (REL / fn).open() as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[2] == "P": G[f[0]].add(alt.get(f[1], f[1]))
    return G

# ---- pool (deployed anchor) ----
esc = pq.read_table(CW / "rerank_out" / "eval_scores.parquet")
ecat = np.asarray(esc.column("category").to_pylist()); easp = np.asarray(esc.column("aspect").to_pylist())
eP = np.asarray(esc.column("protein_accession").to_pylist())
eG = np.array([alt.get(g, g) for g in np.asarray(esc.column("go_term_id").to_pylist())])
eS = esc.column("reranker_score").to_numpy(zero_copy_only=False).astype(np.float64)

# ---- method candidate generators (top-K BP/prot, scored) ----
def learned_topk(variant, prots, K):
    gc = np.load(OUT / f"go_codes_{variant}.npz", allow_pickle=True)
    vocab = gc["go_ids"]; G = torch.from_numpy(gc["codes"].astype(np.float32)).to(dev)
    pj = np.load(OUT / f"eval_proj_codes_{variant}.npz", allow_pickle=True)
    a2r = {a: i for i, a in enumerate(pj["accs"])}; Pc = torch.from_numpy(pj["codes"].astype(np.float32)).to(dev)
    keep = [p for p in prots if p in a2r]; rows = torch.tensor([a2r[p] for p in keep])
    out = {}
    with torch.no_grad():
        for s in range(0, len(keep), 256):
            sim = Pc[rows[s:s+256].to(dev)] @ G.t()
            v, idx = torch.topk(sim, K, dim=1); v = v.cpu().numpy(); idx = idx.cpu().numpy()
            for r, p in enumerate(keep[s:s+256]):
                out[p] = list(zip([alt.get(x, x) for x in vocab[idx[r]]], v[r].tolist()))
    return out

def dense_topk(prots, K):
    dA = np.load(CW / "seq2ann_infonce" / "eval_scores_armA.npz", allow_pickle=True)
    by = collections.defaultdict(list)
    for p_, t_, s_ in zip(dA["prot"], dA["term"], dA["score"]): by[p_].append((alt.get(t_, t_), float(s_)))
    out = {}
    for p in prots:
        if p in by:
            out[p] = sorted(by[p], key=lambda x: -x[1])[:K]
    return out

def run_cell(cell, gt_fn, known):
    m = (ecat == cell) & (easp == "bpo")
    P, Gc, Sc = eP[m], eG[m], eS[m]
    pool = collections.defaultdict(dict)
    for p_, g_, s_ in zip(P, Gc, Sc): pool[p_][g_] = s_
    GTraw = load_gt(gt_fn)
    prots = sorted(set(GTraw) & set(pool))
    GT = {p: GTraw[p] for p in prots}
    base = [(p_, g_, float(s_)) for p_, g_, s_ in zip(P, Gc, Sc) if p_ in GT]
    pool_scores = np.array([r[2] for r in base])          # deployed score distribution
    A = cafa(base, GT, known)
    log(f"{cell} anchor A (pool, true frame) = {A}  prots={len(prots)}")
    out = {"anchor_A": A, "n_prots": len(prots),
           "note": "extra scores QUANTILE-CALIBRATED onto the pool reranker_score distribution "
                   "(pool range %.2f..%.2f); top-half by confidence submitted; identical for every arm; "
                   "FIXEDSCORE control = random confidence." % (pool_scores.min(), pool_scores.max()),
           "arms": {}}
    rng = np.random.default_rng(0)
    def calibrate_and_submit(ep, eg, es):
        """Map extra confidence -> pool score quantile; submit the upper-half-confidence extras."""
        es = np.asarray(es, float)
        order = np.argsort(np.argsort(es)); q = (order + 0.5) / len(es)      # confidence quantile
        mapped = np.quantile(pool_scores, q)
        return [(p_, g_, float(v)) for p_, g_, v, qq in zip(ep, eg, mapped, q) if qq > 0.5]
    gens = {"LEARNED": learned_topk(LEARNED, prots, 200),
            "TWOTOWER": learned_topk("twotower", prots, 200),
            "DENSE": dense_topk(prots, 200)}
    for K in (100, 200):
        for name, gen in gens.items():
            ep, eg, es = [], [], []
            for p in prots:
                pl = pool.get(p, {})
                for g_, sc in gen.get(p, [])[:K]:
                    if g_ in BP and g_ not in pl: ep.append(p); eg.append(g_); es.append(sc)
            if len(es) == 0: continue
            extras = calibrate_and_submit(ep, eg, es)
            f = cafa(base + extras, GT, known)
            out["arms"].setdefault(f"top{K}", {})[name] = f
            out["arms"][f"top{K}"][f"{name}_delta"] = round(f - A, 5) if (f is not None and A) else None
            if name == "LEARNED":   # fixed-score control on the SAME learned candidates
                exc = calibrate_and_submit(ep, eg, rng.random(len(ep)))
                fc = cafa(base + exc, GT, known)
                out["arms"][f"top{K}"]["FIXEDSCORE_control"] = fc
                out["arms"][f"top{K}"]["FIXEDSCORE_control_delta"] = round(fc - A, 5) if (fc is not None and A) else None
            log(f"{cell} top{K} {name}: f={out['arms'][f'top{K}'].get(name)} "
                f"delta={out['arms'][f'top{K}'].get(name+'_delta')} extras={len(extras)}")
        json.dump(RES, open(OUT / "phase3_fmicrow.json", "w"), indent=1)
    return out

RES = {"frame": "TRUE board frame; prop=fill norm=cafa no_orphans toi; PK -known (evaluation.nf:279)",
       "noise_floor": 0.0034, "learned_variant": LEARNED, "cells": {}}
RES["cells"]["PK_BP"] = run_cell("pk", "groundtruth_PK.tsv", str(REL / "groundtruth_PK_known.tsv"))
RES["cells"]["LK_BP"] = run_cell("lk", "groundtruth_LK.tsv", None)
json.dump(RES, open(OUT / "phase3_fmicrow.json", "w"), indent=1)
log("DONE fmicrow decider")
