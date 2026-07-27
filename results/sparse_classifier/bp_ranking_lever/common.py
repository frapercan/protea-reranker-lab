"""Shared helpers for the BP term-space ranking lever."""
import os, json
import numpy as np
import pyarrow.parquet as pq

OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA_PATH = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
NPZ = "/home/frapercan/Thesis2/storage/two_tower_sparse/per_cut/go_sparse_codes_v227.npz"
PARENTS = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/go_parents.json"

PERCUT = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
BASELINE_DIR = os.path.join(PERCUT, "baseline")

# base prediction sources (the graft's per-category BP sources)
LK_PRED = os.path.join(BASELINE_DIR, "predictions/lk/lk.tsv")
PK_PRED = os.path.join(PERCUT, "predictions/pk/pk.tsv")
LK_MODEL = os.path.join(BASELINE_DIR, "predictions/model_lk.txt")
PK_MODEL = os.path.join(PERCUT, "predictions/model_pk.txt")
LK_TRAIN_PARQUET = os.path.join(BASELINE_DIR, "train.parquet")
PK_TRAIN_PARQUET = os.path.join(PERCUT, "train.parquet")


def load_obo_namespace(path=OBO):
    """GO id -> namespace (biological_process/molecular_function/cellular_component)."""
    ns = {}
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = None
            elif line.startswith("id: GO:"):
                cur = line[4:]
            elif line.startswith("namespace:") and cur:
                ns[cur] = line.split(" ", 1)[1].strip()
    return ns


def load_ia(path=IA_PATH):
    ia = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    ia[p[0]] = float(p[1])
                except ValueError:
                    pass
    return ia


def load_codes(path=NPZ):
    """Return go_id->row index and L2-normalized code matrix."""
    d = np.load(path, allow_pickle=True)
    go_ids = d["go_ids"]
    codes = d["codes"].astype(np.float32)
    nrm = np.linalg.norm(codes, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    codes = codes / nrm
    idx = {g: i for i, g in enumerate(go_ids)}
    return idx, codes


def compute_depths(parents_path=PARENTS):
    """Longest path to a root (closure-based depth) for each GO term."""
    parents = json.load(open(parents_path))
    depth = {}

    def d(go):
        if go in depth:
            return depth[go]
        ps = parents.get(go, [])
        if not ps:
            depth[go] = 0
            return 0
        depth[go] = -1  # cycle guard
        best = 0
        for p in ps:
            if depth.get(p, 0) == -1:
                continue
            best = max(best, 1 + d(p))
        depth[go] = best
        return best

    import sys
    sys.setrecursionlimit(100000)
    for g in parents:
        d(g)
    return depth


# ---- reranker feature reconstruction (to score validation rows with saved model) ----
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}


def feature_cols(parquet_path):
    schema = pq.ParquetFile(parquet_path).schema_arrow
    all_cols = list(schema.names)
    bool_cols = [n for n in all_cols if str(schema.field(n).type) == "bool"]
    feats = [c for c in all_cols if c not in META] + ["aspect_code"]
    return feats, bool_cols, all_cols


def load_valid_bp(parquet_path, category):
    """Load v225-v227 BP rows for a category, with feature matrix ready for booster."""
    feats, bool_cols, all_cols = feature_cols(parquet_path)
    read_cols = list(META) + [c for c in feats if c != "aspect_code"]
    filt = [("category", "=", category), ("aspect", "=", "bpo"),
            ("snapshot_pair", "=", "v225-v227")]
    if category == "pk":
        filt.append(("knn_present", "=", True))
    df = pq.read_table(parquet_path, columns=read_cols, filters=filt).to_pandas()
    for b in bool_cols:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in feats:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df, feats
