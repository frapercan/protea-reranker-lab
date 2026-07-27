"""NK-BPO leave-one-family-out (LOFO) receipt on the clean 227->230 frame.

Trains the NK per-category re-ranker (all NK aspects, binary label) on the full
69-feature schema, then retrains dropping each feature family and re-measures the
board-faithful NK-BPO f_micro_w. delta = baseline - dropped  (positive => the
family HELPS NK-BPO). This is the signal-by-signal receipt for "where NK-BPO #1
comes from". Train negatives subsampled (neg_frac); eval is full (exact cells).
"""
import os, json, time
import numpy as np, pandas as pd
import pyarrow.dataset as pads, pyarrow.compute as pc, pyarrow.parquet as pq
import lightgbm as lgb
from cafaeval.evaluation import cafa_eval
from protea_contracts.feature_schema import FEATURE_FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
PCd = os.path.join(SC, "percut_rerank")
BASE = os.path.join(PCd, "baseline")
PRED = os.path.join(HERE, "pred"); os.makedirs(PRED, exist_ok=True)
SEED = 42; NEG_FRAC = 0.15
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
_sch = pq.ParquetFile(os.path.join(BASE, "train.parquet")).schema_arrow
ALL_COLS = list(_sch.names)
BOOL_COLS = [n for n in ALL_COLS if str(_sch.field(n).type) == "bool"]
FEATS = [c for c in ALL_COLS if c not in META]
READ = list(META) + FEATS

# families: FEATURE_FAMILIES present + own-groups for the LAFA signals
def build_families():
    cols = set(FEATS)
    fam = {}
    for k, v in FEATURE_FAMILIES.items():
        present = [c for c in v if c in cols]
        if present:
            fam[k] = present
    fam["interpro"] = [c for c in FEATS if c.startswith("interpro")]
    fam["classifier"] = [c for c in FEATS if c.startswith("classifier")]
    fam["association"] = [c for c in FEATS if c.startswith("association")]
    fam["self_prior"] = [c for c in FEATS if c.startswith("self_prior")]
    return {k: v for k, v in fam.items() if v}
FAMS = build_families()

GT_DIR = os.path.join(SC, "lafa_gt")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = os.path.join(GT_DIR, "groundtruth_terms_of_interest.txt")
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
GT_NK = os.path.join(GT_DIR, "groundtruth_NK.tsv")


def load(path, neg_frac=1.0):
    dset = pads.dataset(path, format="parquet")
    sc = dset.scanner(columns=READ, filter=(pc.field("category") == "nk"), batch_size=400_000)
    rng = np.random.default_rng(SEED); parts = []
    for rb in sc.to_batches():
        if not rb.num_rows:
            continue
        df = rb.to_pandas()
        if neg_frac < 1.0 and len(df):
            neg = df[df.label == 0]
            keep = neg.sample(frac=neg_frac, random_state=int(rng.integers(1 << 30))) if len(neg) else neg
            df = pd.concat([df[df.label == 1], keep])
        if len(df):
            parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=READ)
    for b in BOOL_COLS:
        if b in out.columns:
            out[b] = out[b].astype("int8")
    for c in FEATS:
        if c in out.columns and out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out


def nk_bpo(model, ev, feats, tag):
    ev = ev.copy()
    ev["score"] = model.predict(ev[feats], num_iteration=model.best_iteration)
    d = os.path.join(PRED, tag); os.makedirs(d, exist_ok=True)
    ev[["protein_accession", "go_term_id", "score"]].to_csv(
        os.path.join(d, "nk.tsv"), sep="\t", header=False, index=False, float_format="%.6f")
    _, best = cafa_eval(OBO, d, GT_NK, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                        toi_file=TOI, th_step=0.01, n_cpu=8)
    for _, r in best["f_micro_w"].reset_index().iterrows():
        if NS2ASP.get(r["ns"]) == "bpo":
            return round(float(r["f_micro_w"]), 5)
    return None


def train(tr, va, feats):
    dtr = lgb.Dataset(tr[feats], label=tr.label.astype("int8").values, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va.label.astype("int8").values, reference=dtr, free_raw_data=False)
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
             min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, seed=SEED, verbosity=-1, num_threads=12)
    return lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva], valid_names=["v"],
                     callbacks=[lgb.early_stopping(60, verbose=False)])


def main():
    t0 = time.time()
    full = load(os.path.join(BASE, "train.parquet"), NEG_FRAC)
    ev = load(os.path.join(BASE, "eval.parquet"), 1.0)
    tr = full[full.snapshot_pair != "v225-v227"].copy()
    va = full[full.snapshot_pair == "v225-v227"].copy()
    print(f"[load] train {len(tr)} pos {int(tr.label.sum())} | valid {len(va)} | eval {len(ev)} "
          f"({time.time()-t0:.0f}s) | {len(FAMS)} families", flush=True)
    base_model = train(tr, va, FEATS)
    base = nk_bpo(base_model, ev, FEATS, "baseline")
    print(f"[baseline] NK-BPO f_micro_w = {base}  ({time.time()-t0:.0f}s)", flush=True)
    rows = []
    for i, (fam, cols) in enumerate(sorted(FAMS.items()), 1):
        feats = [c for c in FEATS if c not in set(cols)]
        m = train(tr, va, feats)
        v = nk_bpo(m, ev, feats, f"drop_{fam}")
        rows.append({"family": fam, "n_cols": len(cols), "nk_bpo": v,
                     "delta": round(base - v, 5) if v is not None else None})
        print(f"[{i}/{len(FAMS)}] drop {fam:16s} ({len(cols)} cols): NK-BPO={v} "
              f"delta={base-v:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: -(r["delta"] or -9))
    out = {"baseline_nk_bpo": base, "neg_frac": NEG_FRAC,
           "lofo": rows, "note": "delta = baseline - dropped; positive => family helps NK-BPO"}
    json.dump(out, open(os.path.join(HERE, "result.json"), "w"), indent=2)
    print("\n=== NK-BPO LOFO (sorted, most important first) ===", flush=True)
    for r in rows:
        print(f"  {r['family']:16s} delta {r['delta']:+.5f}  (drop -> {r['nk_bpo']})", flush=True)
    print(f"\nwrote result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
