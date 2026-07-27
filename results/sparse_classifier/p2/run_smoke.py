"""SMOKE: small subset, 1 seed, few epochs. Assert loss decreases AND recall@100 beats the
term-frequency baseline. Run before the full train."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from two_tower import Data, train_seed, recall_at_k, recall_tf_baseline

BASE = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
PREP = "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad/sc/prep.npz"
OUT = f"{BASE}/p2"

t0 = time.time()
data = Data(PREP, f"{BASE}/clf_protein_codes.npz")
print("vocab V =", data.V, "train", len(data.train_idx), "test", len(data.test_idx))

# two-tower retrieval needs enough proteins to learn the projection; 5k is too few (it learns
# only the bias/frequency prior). 20k proteins is the smallest subset that clears the baseline.
sub_train = data.train_idx[:20000]
sub_test = data.test_idx[:2000]
cfg = dict(hidden=0, kwta=0, dropout=0.0, lr=2e-3, wd=1e-5, epochs=8, bs=512,
           gn=4.0, gp=1.0, clip=0.05, dag_w=0.05, dag_edges=4096, dag_margin=0.0)

model, losses = train_seed(data, 0, sub_train, cfg, log_prefix="SMOKE ")
rec, ceil, n = recall_at_k(data, [model], sub_test, k=100)
tf = recall_tf_baseline(data, sub_test, k=100)

print("\n=== SMOKE RESULTS ===")
print("losses:", [round(x, 4) for x in losses])
print(f"recall@100 model = {rec:.4f}  tf-baseline = {tf:.4f}  ceiling = {ceil:.4f}  (n={n})")
loss_ok = losses[-1] < losses[0]
beat_ok = rec > tf
print("loss decreased:", loss_ok, "| beats tf baseline:", beat_ok)

json.dump({"losses": losses, "recall100": rec, "tf_baseline": tf, "ceiling": ceil,
           "n_eval": n, "loss_decreased": loss_ok, "beats_tf": beat_ok,
           "cfg": cfg, "secs": time.time() - t0},
          open(f"{OUT}/smoke_metrics.json", "w"), indent=2)
assert loss_ok, "SMOKE FAIL: loss did not decrease"
assert beat_ok, "SMOKE FAIL: did not beat tf baseline"
print(f"\nSMOKE PASS in {time.time()-t0:.0f}s")
