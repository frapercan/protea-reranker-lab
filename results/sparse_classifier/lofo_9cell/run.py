"""9-cell leave-one-family-out (LOFO) master ablation, board-faithful.

For each category (nk, lk, pk) trains the per-category re-ranker on the clean
227->230 frame (nk/lk = baseline pool; pk = percut pool, knn_present) on the full
69-feature schema, then retrains dropping each feature family and re-measures the
board-faithful f_micro_w for all three aspects (mfo/bpo/cco). delta = baseline -
dropped (positive => family helps that cell). LK-BPO gets the champion per-protein
pmin-pmax; PK excludes PK_known. Train negatives subsampled; eval is full.
"""
import os, json, time
import numpy as np, pandas as pd
import pyarrow.dataset as pads, pyarrow.compute as pc, pyarrow.parquet as pq
import lightgbm as lgb
from cafaeval.evaluation import cafa_eval
from protea_contracts.feature_schema import FEATURE_FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
PCd = os.path.join(SC, "percut_rerank"); BASE = os.path.join(PCd, "baseline")
PRED = os.path.join(HERE, "pred"); os.makedirs(PRED, exist_ok=True)
SEED = 42; NEG_FRAC = 0.15
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
_sch = pq.ParquetFile(os.path.join(BASE, "train.parquet")).schema_arrow
ALL_COLS = list(_sch.names); BOOL_COLS = [n for n in ALL_COLS if str(_sch.field(n).type) == "bool"]
FEATS = [c for c in ALL_COLS if c not in META]; READ = list(META) + FEATS
NS2ASP = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
GT = SC + "/lafa_gt"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = GT + "/groundtruth_terms_of_interest.txt"
# category -> (pool_dir, gt_file, exclude_file)
CATS = {"nk": (BASE, GT + "/groundtruth_NK.tsv", None),
        "lk": (BASE, GT + "/groundtruth_LK.tsv", None),
        "pk": (PCd, GT + "/groundtruth_PK.tsv", GT + "/groundtruth_PK_known.tsv")}


def build_families():
    cols = set(FEATS); fam = {}
    for k, v in FEATURE_FAMILIES.items():
        p = [c for c in v if c in cols]
        if p:
            fam[k] = p
    for k, pre in [("interpro", "interpro"), ("classifier", "classifier"),
                   ("association", "association"), ("self_prior", "self_prior")]:
        c = [x for x in FEATS if x.startswith(pre)]
        if c:
            fam[k] = c
    return fam
FAMS = build_families()


def load(cat, split, neg_frac=1.0):
    path = os.path.join(CATS[cat][0], f"{split}.parquet")
    sc = pads.dataset(path, format="parquet").scanner(
        columns=READ, filter=(pc.field("category") == cat), batch_size=400_000)
    rng = np.random.default_rng(SEED); parts = []
    for rb in sc.to_batches():
        if not rb.num_rows:
            continue
        df = rb.to_pandas()
        if cat == "pk" and "knn_present" in df.columns:
            df = df[df["knn_present"].astype(bool)]
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


def pminmax_bpo(out):
    m = out["aspect"] == "bpo"
    if not m.any():
        return out
    sub = out.loc[m, ["protein_accession", "score"]]
    g = sub.groupby("protein_accession")["score"]
    mn, mx = g.transform("min"), g.transform("max"); rng = mx - mn
    out.loc[m, "score"] = np.where(rng > 0, (sub["score"] - mn) / rng, 1.0)
    return out


def train(tr, va, feats):
    dtr = lgb.Dataset(tr[feats], label=tr.label.astype("int8").values, free_raw_data=False)
    dva = lgb.Dataset(va[feats], label=va.label.astype("int8").values, reference=dtr, free_raw_data=False)
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
             min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, seed=SEED, verbosity=-1, num_threads=12)
    return lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva], valid_names=["v"],
                     callbacks=[lgb.early_stopping(60, verbose=False)])


def eval_cells(model, ev, feats, cat, tag):
    ev = ev.copy()
    ev["score"] = model.predict(ev[feats], num_iteration=model.best_iteration)
    if cat == "lk":
        ev = pminmax_bpo(ev)
    d = os.path.join(PRED, f"{cat}_{tag}"); os.makedirs(d, exist_ok=True)
    ev[["protein_accession", "go_term_id", "score"]].to_csv(
        os.path.join(d, f"{cat}.tsv"), sep="\t", header=False, index=False, float_format="%.6f")
    excl = CATS[cat][2]
    _, best = cafa_eval(OBO, d, CATS[cat][1], ia=IA, no_orphans=True, norm="cafa",
                        prop="fill", exclude=excl, toi_file=TOI, th_step=0.01, n_cpu=8)
    cells = {}
    for _, r in best["f_micro_w"].reset_index().iterrows():
        a = NS2ASP.get(r["ns"])
        if a:
            cells[a] = round(float(r["f_micro_w"]), 5)
    return cells


def main():
    t0 = time.time(); result = {"baseline": {}, "lofo": {}}
    for cat in ("nk", "lk", "pk"):
        full = load(cat, "train", NEG_FRAC); ev = load(cat, "eval", 1.0)
        tr = full[full.snapshot_pair != "v225-v227"].copy()
        va = full[full.snapshot_pair == "v225-v227"].copy()
        print(f"[{cat}] train {len(tr)} pos {int(tr.label.sum())} eval {len(ev)} ({time.time()-t0:.0f}s)", flush=True)
        bm = train(tr, va, FEATS); bcells = eval_cells(bm, ev, FEATS, cat, "baseline")
        result["baseline"][cat] = bcells
        print(f"[{cat}] baseline {bcells} ({time.time()-t0:.0f}s)", flush=True)
        result["lofo"][cat] = {}
        for fam, cols in sorted(FAMS.items()):
            feats = [c for c in FEATS if c not in set(cols)]
            m = train(tr, va, feats); cells = eval_cells(m, ev, feats, cat, f"drop_{fam}")
            result["lofo"][cat][fam] = {a: round(bcells[a] - cells.get(a, 0), 5) for a in bcells}
            print(f"  [{cat}] drop {fam:16s} bpo_delta={bcells.get('bpo',0)-cells.get('bpo',0):+.5f} "
                  f"mfo={bcells.get('mfo',0)-cells.get('mfo',0):+.5f} cco={bcells.get('cco',0)-cells.get('cco',0):+.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del full, tr, va, ev
    json.dump(result, open(os.path.join(HERE, "result.json"), "w"), indent=2)
    # print master: dominant family per cell
    print("\n=== 9-CELL DOMINANT SIGNAL (top family by delta per cell) ===", flush=True)
    for cat in ("nk", "lk", "pk"):
        for a in ("mfo", "bpo", "cco"):
            deltas = [(f, result["lofo"][cat][f][a]) for f in result["lofo"][cat]]
            deltas.sort(key=lambda x: -x[1]); top = deltas[:3]
            base = result["baseline"][cat][a]
            print(f"  {cat}-{a} (base {base}): " + ", ".join(f"{f} {d:+.4f}" for f, d in top), flush=True)
    print(f"\nwrote result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
