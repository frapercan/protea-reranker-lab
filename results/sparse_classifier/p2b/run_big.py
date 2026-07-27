"""P2b FULL: 7-seed ensemble two-tower sparse functional candidate generator on the
LARGE CURATED v227 corpus (~520K proteins). Same SEEDS + CFG as p2/run_full.py.
Saves per-seed ckpts to p2b/, evaluates top-k recall vs term-frequency baseline +
ceiling on the held-out 10% protein split, stratified by #positives. Writes
p2b/metrics.json. No live DB; pure file -> file.
"""
import os, sys, json, time, numpy as np, torch
P2B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(P2B), "p2"))
from two_tower import (Data, train_seed, recall_at_k, recall_tf_baseline, DEV)

BASE = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
OUT = P2B
PREP = f"{OUT}/prep_big.npz"
SEEDS = [0, 7, 137, 23, 91, 31, 53]
CFG = dict(hidden=2048, kwta=0, dropout=0.0, lr=2e-3, wd=1e-5, epochs=20, bs=512,
           gn=4.0, gp=1.0, clip=0.05, dag_w=0.2, dag_edges=4096, dag_margin=0.0)

t0 = time.time()
data = Data(PREP, f"{BASE}/clf_protein_codes_big.npz")
print(f"vocab V={data.V} train={len(data.train_idx)} test={len(data.test_idx)} dev={DEV}", flush=True)
print(f"codes on {data.codes.device} {tuple(data.codes.shape)} "
      f"GPUmem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

models, all_losses = [], {}
for sd in SEEDS:
    m, losses = train_seed(data, sd, data.train_idx, CFG, log_prefix="BIG ")
    torch.save({"state_dict": m.state_dict(), "cfg": CFG, "V": data.V, "seed": sd},
               f"{OUT}/head_seed{sd}.pt")
    rec, ceil, n = recall_at_k(data, [m], data.test_idx, 100)
    print(f"  seed{sd} solo recall@100={rec:.4f}  "
          f"GPUmem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    all_losses[sd] = losses
    models.append(m)

tf100 = recall_tf_baseline(data, data.test_idx, 100)
metrics = {"corpus": "520K-curated-v227", "vocab_size": int(data.V),
           "n_train": int(len(data.train_idx)), "n_test": int(len(data.test_idx)),
           "seeds": SEEDS, "cfg": CFG, "losses_per_seed": all_losses,
           "tf_baseline_recall100": tf100, "recall_at_k": {}, "solo_recall100": {}}
for k in [50, 100, 200]:
    rec, ceil, n = recall_at_k(data, models, data.test_idx, k)
    tfk = recall_tf_baseline(data, data.test_idx, k)
    metrics["recall_at_k"][str(k)] = {"ensemble": rec, "ceiling": ceil,
                                      "tf_baseline": tfk, "n": n}
    print(f"k={k}: ensemble={rec:.4f}  tf={tfk:.4f}  ceiling={ceil:.4f}", flush=True)


@torch.no_grad()
def stratified(models, eval_idx, k=100, bs=512):
    GO_T = data.GO_V.t().contiguous()
    buckets = {"1-10": [], "11-30": [], "31-60": [], "61-100": [], ">100": []}
    for s in range(0, len(eval_idx), bs):
        bidx = eval_idx[s:s + bs]; x = data.codes[bidx]
        scores = None
        for m in models:
            m.eval(); scores = m(x, GO_T) if scores is None else scores + m(x, GO_T)
        top = scores.topk(k, dim=1).indices.cpu().numpy()
        for b, i in enumerate(bidx):
            pos = set(data.positives(i).tolist())
            if not pos:
                continue
            r = len(pos & set(top[b].tolist())) / len(pos); np_ = len(pos)
            key = ("1-10" if np_ <= 10 else "11-30" if np_ <= 30 else "31-60" if np_ <= 60
                   else "61-100" if np_ <= 100 else ">100")
            buckets[key].append(r)
    return {k2: {"mean_recall": float(np.mean(v)), "n": len(v)}
            for k2, v in buckets.items() if v}


metrics["stratified_by_npos_recall100"] = stratified(models, data.test_idx, 100)
metrics["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
metrics["secs"] = time.time() - t0
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)
print("\nstratified@100:", json.dumps(metrics["stratified_by_npos_recall100"], indent=2))
print(f"\nDONE in {time.time()-t0:.0f}s. metrics -> {OUT}/metrics.json", flush=True)
