"""Train the joint protein<->term scorer with the LEARNABLE GCN GO-label encoder.

Joint model = ProjHead (protein tower, reused from p2/two_tower.py) + LabelGCN (label
tower, label_gcn.py). Each step recomputes the DAG-aware term embeddings over the full
vocab and scores the batch against them; both towers + temperature + bias train end to
end with ASL + soft DAG-consistency hinge (same losses as the frozen two-tower).

Run:
  python train_gcn.py smoke      # tiny subset, 1 seed, asserts loss-down + beats tf
  python train_gcn.py full       # full train split, N seeds, saves ckpts + recall eval

Trained on the v227 prep.npz snapshot (the test t0), mirroring the frozen two-tower
protocol; per-cut honest scoring happens at inference (gcn_scorer.py).
"""
import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "p2"))
sys.path.insert(0, HERE)
from two_tower import Data, ProjHead, asl_loss, dag_hinge, recall_tf_baseline, DEV  # noqa: E402
from label_gcn import LabelGCN, build_norm_adj  # noqa: E402

PREP = os.path.join(SC, "p2", "prep.npz")
PROT_CODES = os.path.join(SC, "clf_protein_codes.npz")


def make_models(data, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    proj = ProjHead(2048, 1024, V=data.V, hidden=cfg["hidden"], kwta=cfg["kwta"],
                    dropout=cfg["dropout"]).to(DEV)
    gcn = LabelGCN(d_in=1024, d_hid=cfg["gcn_hid"], d_out=1024,
                   layers=cfg["gcn_layers"], dropout=cfg["gcn_dropout"]).to(DEV)
    return proj, gcn


def build_target(data, batch_idx):
    B = len(batch_idx)
    tgt = torch.zeros(B, data.V, device=DEV)
    for b, i in enumerate(batch_idx):
        pos = data.positives(i)
        if len(pos):
            tgt[b, pos] = 1.0
    return tgt


def train_seed(data, A, init, seed, train_idx, cfg, log_prefix=""):
    proj, gcn = make_models(data, cfg, seed)
    params = list(proj.parameters()) + list(gcn.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    ec = data.edges[0].to(DEV); ep = data.edges[1].to(DEV); n_edge = ec.shape[0]
    losses = []
    for ep_i in range(cfg["epochs"]):
        proj.train(); gcn.train()
        rng = np.random.default_rng(seed * 1000 + ep_i)
        order = rng.permutation(len(train_idx))
        tot = 0.0; nb = 0
        for s in range(0, len(order), cfg["bs"]):
            bidx = train_idx[order[s:s + cfg["bs"]]]
            x = data.codes[bidx]
            tgt = build_target(data, bidx)
            gcn_emb = gcn(init, A)                 # V x 1024 (recomputed each step)
            logits = proj(x, gcn_emb.t().contiguous())
            loss = asl_loss(logits, tgt, cfg["gn"], cfg["gp"], cfg["clip"])
            if cfg["dag_w"] > 0 and n_edge > 0:
                ne = min(cfg["dag_edges"], n_edge)
                sel = torch.randint(0, n_edge, (ne,), device=DEV)
                loss = loss + cfg["dag_w"] * dag_hinge(logits, ec[sel], ep[sel], cfg["dag_margin"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        avg = tot / max(nb, 1); losses.append(avg)
        print(f"{log_prefix}seed{seed} ep{ep_i} loss {avg:.4f}", flush=True)
    return proj, gcn, losses


@torch.no_grad()
def recall_at_k(data, ensemble, A, init, eval_idx, k=100, bs=512):
    """ensemble: list of (proj, gcn). Averaged ensemble logits, mean top-k recall."""
    embs = []
    for proj, gcn in ensemble:
        proj.eval(); gcn.eval()
        embs.append(gcn(init, A).t().contiguous())
    rec = []; capped = []
    for s in range(0, len(eval_idx), bs):
        bidx = eval_idx[s:s + bs]
        x = data.codes[bidx]
        scores = None
        for (proj, _), GO_T in zip(ensemble, embs):
            lg = proj(x, GO_T)
            scores = lg if scores is None else scores + lg
        topk = scores.topk(k, dim=1).indices.cpu().numpy()
        for b, i in enumerate(bidx):
            pos = set(data.positives(i).tolist())
            if not pos:
                continue
            rec.append(len(pos & set(topk[b].tolist())) / len(pos))
            capped.append(min(k, len(pos)) / len(pos))
    return float(np.mean(rec)), float(np.mean(capped)), len(rec)


def load_data_and_graph():
    data = Data(PREP, PROT_CODES)
    init = data.GO_V.clone()                      # V x 1024 (frozen v227 fused codes)
    A = build_norm_adj(data.edges.cpu().numpy(), data.V, DEV)
    return data, init, A


SMOKE_CFG = dict(hidden=0, kwta=0, dropout=0.0, lr=2e-3, wd=1e-5, epochs=8, bs=512,
                 gn=4.0, gp=1.0, clip=0.05, dag_w=0.05, dag_edges=4096, dag_margin=0.0,
                 gcn_hid=1024, gcn_layers=2, gcn_dropout=0.0)
FULL_CFG = dict(hidden=2048, kwta=0, dropout=0.0, lr=2e-3, wd=1e-5, epochs=20, bs=512,
                gn=4.0, gp=1.0, clip=0.05, dag_w=0.2, dag_edges=4096, dag_margin=0.0,
                gcn_hid=1024, gcn_layers=2, gcn_dropout=0.0)
SEEDS = [0, 7, 23]


def run_smoke():
    t0 = time.time()
    data, init, A = load_data_and_graph()
    print(f"vocab V={data.V} train={len(data.train_idx)} test={len(data.test_idx)} dev={DEV}", flush=True)
    sub_train = data.train_idx[:20000]
    sub_test = data.test_idx[:2000]
    proj, gcn, losses = train_seed(data, A, init, 0, sub_train, SMOKE_CFG, "SMOKE ")
    rec, ceil, n = recall_at_k(data, [(proj, gcn)], A, init, sub_test, 100)
    tf = recall_tf_baseline(data, sub_test, 100)
    loss_ok = losses[-1] < losses[0]; beat_ok = rec > tf
    out = {"losses": losses, "recall100": rec, "tf_baseline": tf, "ceiling": ceil,
           "n_eval": n, "loss_decreased": loss_ok, "beats_tf": beat_ok,
           "cfg": SMOKE_CFG, "secs": time.time() - t0}
    json.dump(out, open(os.path.join(HERE, "smoke_metrics.json"), "w"), indent=2)
    print(f"\nSMOKE losses {[round(x,3) for x in losses]}")
    print(f"recall@100 model={rec:.4f} tf={tf:.4f} ceiling={ceil:.4f} n={n}")
    print("loss decreased:", loss_ok, "| beats tf:", beat_ok, flush=True)
    assert loss_ok, "SMOKE FAIL: loss did not decrease"
    assert beat_ok, "SMOKE FAIL: did not beat tf baseline"
    print(f"SMOKE PASS in {time.time()-t0:.0f}s", flush=True)


def run_full():
    t0 = time.time()
    data, init, A = load_data_and_graph()
    print(f"vocab V={data.V} train={len(data.train_idx)} test={len(data.test_idx)} dev={DEV}", flush=True)
    ensemble = []; all_losses = {}
    for sd in SEEDS:
        proj, gcn, losses = train_seed(data, A, init, sd, data.train_idx, FULL_CFG, "FULL ")
        torch.save({"proj": proj.state_dict(), "gcn": gcn.state_dict(),
                    "cfg": FULL_CFG, "V": data.V, "seed": sd},
                   os.path.join(HERE, f"gcn_seed{sd}.pt"))
        rec, ceil, n = recall_at_k(data, [(proj, gcn)], A, init, data.test_idx, 100)
        print(f"  seed{sd} solo recall@100={rec:.4f} GPUmem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        ensemble.append((proj, gcn)); all_losses[sd] = losses
    metrics = {"vocab_size": int(data.V), "n_train": int(len(data.train_idx)),
               "n_test": int(len(data.test_idx)), "seeds": SEEDS, "cfg": FULL_CFG,
               "losses_per_seed": all_losses, "recall_at_k": {}}
    for k in [50, 100, 200]:
        rec, ceil, n = recall_at_k(data, ensemble, A, init, data.test_idx, k)
        tfk = recall_tf_baseline(data, data.test_idx, k)
        metrics["recall_at_k"][str(k)] = {"ensemble": rec, "ceiling": ceil, "tf_baseline": tfk, "n": n}
        print(f"k={k}: ensemble={rec:.4f} tf={tfk:.4f} ceiling={ceil:.4f}", flush=True)
    # compare to the frozen two-tower ensemble (p2/metrics.json)
    try:
        tt = json.load(open(os.path.join(SC, "p2", "metrics.json")))
        metrics["two_tower_recall_at_k"] = {k: tt["recall_at_k"][k]["ensemble"] for k in ["50", "100", "200"]}
    except Exception:
        pass
    metrics["secs"] = time.time() - t0
    json.dump(metrics, open(os.path.join(HERE, "gcn_metrics.json"), "w"), indent=2)
    print("recall vs two-tower:", json.dumps(metrics.get("two_tower_recall_at_k", {})), flush=True)
    print(f"DONE full in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        run_smoke()
    else:
        run_full()
