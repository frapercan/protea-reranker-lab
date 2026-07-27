"""Graft TransFew's calibration mechanism onto the deployed per-category reranker (NO new modality).

Mechanism (from BP_SOTA_RESEARCH.md): TransFew's BP edge on IA-weighted-micro-F is credited to
(a) FREQUENCY-PARTITIONED experts (rare terms get their own operating point) + (b) IA-WEIGHTED
calibration.  We graft this at the CALIBRATION layer over our own reranker scores:

  deployed  : raw reranker score, single global tau (cafaeval sweeps tau, takes max f_micro_w)
  plainIA   : ONE isotonic score->P(label), IA-weighted, fit on v<=225. Monotone -> under a global-tau
              max-F this MUST reproduce deployed (control: isolates 'calibration alone').
  freqpart  : SEPARATE isotonic per t0-frequency bin (rare/med/common). Non-monotone ACROSS bins ->
              rare-term pairs can move above frequent-term pairs under the single global tau. This is
              the TransFew frequency-partition mechanism. Only this arm can move the number.

TEMPORAL GATE: bins + isotonic fit on snapshot_pair != v225-v227 (strictly v<=225); applied BLIND to
the v227-v230 deployed predictions.  Frequency bins from reference_annotations v227 (t0 corpus,
structural, not label-derived) -> leakage-safe.

Runs under the PROTEA venv (cafaeval + lightgbm + sklearn).
"""
import os, sys, json, time, collections
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "/home/frapercan/Thesis2/storage/transfew_calib")
import frame

W = "/home/frapercan/Thesis2/storage/transfew_calib"
LAB = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier/percut_rerank"
TRAIN_PARQUET = f"{LAB}/train.parquet"
MODEL = {c: f"{LAB}/predictions/model_{c}.txt" for c in ("lk", "pk")}
DEPLOYED_TSV = {c: f"{LAB}/predictions/{c}/{c}.tsv" for c in ("lk", "pk")}
REF_ANNOT = "/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04/reference_annotations.parquet"
VALID_PAIR = "v225-v227"          # excluded from the fit window (v<=225 only)
ASPECT_CODE = {"mfo": 0, "bpo": 1, "cco": 2}
N_BINS = 3
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


# ---- feature layout, replicated from train_rerank.py -----------------------------------
_schema = pq.ParquetFile(TRAIN_PARQUET).schema_arrow
ALL_COLS = list(_schema.names)
META = {"protein_accession", "go_term_id", "label", "category", "snapshot_pair",
        "qualifier", "evidence_code", "taxonomic_relation", "aspect"}
BOOL_COLS = [n for n in ALL_COLS if str(_schema.field(n).type) == "bool"]
FEATURES = [c for c in ALL_COLS if c not in META] + ["aspect_code"]
READ_COLS = list(META) + [c for c in FEATURES if c != "aspect_code"]


# ---- IA + obo BP graph (via cafaeval, so bins are consistent with the scorer) ----------
IA = {}
for line in open(frame.IA_F):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            IA[p[0]] = float(p[1])
        except ValueError:
            pass

from cafaeval.parser import obo_parser
from cafaeval.graph import _ancestors_csr
onts = obo_parser(frame.OBO, ("is_a", "part_of"), frame.IA_F, False)
ont = onts["biological_process"]
tdict = ont.terms_dict                    # canonical go -> {'index':..}
talt = ont.terms_dict_alt                 # alt_id -> set(canonical)
n_terms = ont.idxs
idx2go = [None] * n_terms
for go, d in tdict.items():
    idx2go[d["index"]] = go


def go2idx(go):
    d = tdict.get(go)
    if d is not None:
        return d["index"]
    s = talt.get(go)
    if s:
        return tdict[next(iter(s))]["index"]
    return -1


BP_GO = set(tdict.keys()) | set(talt.keys())    # every id (canonical+alt) that lives in BP
log(f"BP graph: {n_terms:,} canonical terms, {len(talt):,} alt ids")

# ---- t0 corpus frequency (v227), PROPAGATED, per BP term -------------------------------
# reference_annotations.go_term_id is a PROTEA internal integer id; map to GO accession first.
GOMETA = "/home/frapercan/Thesis2/storage/protea-frozen-v227-2025-09-04/go_term_metadata.parquet"
gm = pq.read_table(GOMETA, columns=["go_term_id", "go_id"])
id2acc = dict(zip(gm.column("go_term_id").to_pylist(), gm.column("go_id").to_pylist()))
indptr, anc = _ancestors_csr(ont)          # ancestors[t] = anc[indptr[t]:indptr[t+1]] (self-incl)
ra = pq.read_table(REF_ANNOT, columns=["accession", "go_term_id"])
racc = np.asarray(ra.column("accession").to_pylist())
rgo = np.asarray(ra.column("go_term_id").to_pylist())
freq = np.zeros(n_terms, dtype=np.int64)
byprot = collections.defaultdict(list)
n_bp_annot = 0
for a_, gid in zip(racc, rgo):
    acc = id2acc.get(gid)
    if acc is None:
        continue
    gi = go2idx(acc)
    if gi >= 0:
        byprot[a_].append(gi)
        n_bp_annot += 1
for a_, gis in byprot.items():
    clo = set()
    for gi in gis:
        clo.update(anc[indptr[gi]:indptr[gi + 1]].tolist())
    for ti in clo:
        freq[ti] += 1
log(f"corpus freq built: {len(byprot):,} proteins with BP annots, {n_bp_annot:,} direct BP annot rows")


def read_cat_bp_fit(cat):
    """Fit-window (v<=225) BP rows for a category: features + label + term + protein + snapshot.
    Replicates load_category (pk -> knn_present only; bool->int8; aspect_code)."""
    filt = [("category", "=", cat), ("aspect", "=", "bpo"),
            ("snapshot_pair", "!=", VALID_PAIR)]
    if cat == "pk":
        filt.append(("knn_present", "=", True))
    t = pq.read_table(TRAIN_PARQUET, columns=READ_COLS, filters=filt)
    df = t.to_pandas()
    for b in BOOL_COLS:
        if b in df.columns:
            df[b] = df[b].astype("int8")
    df["aspect_code"] = df["aspect"].map(ASPECT_CODE).astype("int8")
    for c in FEATURES:
        if c in df.columns and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def booster_scores(df, cat):
    b = lgb.Booster(model_file=MODEL[cat])
    X = df[FEATURES]
    out = np.empty(len(df), dtype=np.float64)
    step = 2_000_000
    for s in range(0, len(df), step):
        out[s:s + step] = b.predict(X.iloc[s:s + step])
    return out


def pminmax_lk(scores, groups):
    """Per-(protein,snapshot) min-max, mirroring apply_pminmax_lk_bpo (LK-BPO only)."""
    out = scores.copy()
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    edges = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        idx = order[a:b]
        v = scores[idx]
        mn, mx = v.min(), v.max()
        out[idx] = (v - mn) / (mx - mn) if mx > mn else 1.0
    return out


# ---- frequency bins: terciles of log-corpus-freq over candidate BP terms ----------------
# Candidate BP terms = terms appearing in the deployed blind predictions + fit-window pool.
def deployed_bp_terms(cat):
    terms = set()
    for line in open(DEPLOYED_TSV[cat]):
        _, g, _ = line.rstrip("\n").split("\t")
        if g in BP_GO:
            terms.add(go2idx(g))
    terms.discard(-1)
    return terms


def main():
    result = {"frame": "TRUE board (prop=fill norm=cafa no_orphans toi; PK -known); metric f_micro_w BP",
              "temporal_gate": "bins+isotonic fit on snapshot_pair!=v225-v227 (v<=225); applied blind to v227-v230",
              "freq_source": "reference_annotations v227, propagated per-BP-term protein count (structural, leakage-safe)",
              "n_bins": N_BINS, "cats": {}}
    calib_store = {}

    for cat in ("lk", "pk"):
        log(f"==== {cat} ====")
        df = read_cat_bp_fit(cat)
        log(f"  fit-window BP rows {len(df):,} pos {int(df.label.sum()):,}")
        sc = booster_scores(df, cat)
        if cat == "lk":
            grp = (df["protein_accession"].astype(str) + "|" + df["snapshot_pair"].astype(str)).values
            sc = pminmax_lk(sc, grp)
        terms = df["go_term_id"].values
        tidx = np.array([go2idx(g) for g in terms])
        lab = df["label"].values.astype(np.float64)
        w = np.array([IA.get(idx2go[ti] if ti >= 0 else "", 0.0) for ti in tidx])

        # bins from candidate terms (blind ∪ fit) freq terciles
        cand = deployed_bp_terms(cat) | set(int(x) for x in np.unique(tidx) if x >= 0)
        cand = np.array(sorted(cand))
        cf = freq[cand].astype(np.float64)
        lf = np.log1p(cf)
        q1, q2 = np.quantile(lf, [1 / 3, 2 / 3])
        # bin edges on log1p(freq)
        def binof(ti):
            if ti < 0:
                return 1
            v = np.log1p(freq[ti])
            return 0 if v <= q1 else (1 if v <= q2 else 2)
        bin_of_idx = {int(ti): binof(int(ti)) for ti in cand}
        rbin = np.array([bin_of_idx.get(int(ti), 1) for ti in tidx])

        # fit isotonic per bin (freqpart) and one global (plainIA); IA-weighted, IA>0 rows only
        m = w > 0
        iso_bins = {}
        for bnum in range(N_BINS):
            sel = m & (rbin == bnum)
            if sel.sum() < 50 or lab[sel].sum() < 5:
                iso_bins[bnum] = None
                continue
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(sc[sel], lab[sel], sample_weight=w[sel])
            iso_bins[bnum] = ir
        ir_all = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir_all.fit(sc[m], lab[m], sample_weight=w[m])

        calib_store[cat] = {"iso_bins": iso_bins, "ir_all": ir_all,
                            "bin_of_idx": bin_of_idx, "q1": float(q1), "q2": float(q2)}

        # bin diagnostics
        bstats = {}
        for bnum in range(N_BINS):
            sel = rbin == bnum
            bstats[bnum] = {"fit_rows": int(sel.sum()), "fit_pos": int(lab[sel].sum()),
                            "n_cand_terms": int(sum(1 for v in bin_of_idx.values() if v == bnum)),
                            "base_rate": float(lab[sel].mean()) if sel.any() else None,
                            "fit_ok": iso_bins[bnum] is not None}
        result["cats"][cat] = {"fit_rows": int(len(df)), "fit_pos": int(df.label.sum()),
                               "freq_terciles_log1p": [float(q1), float(q2)],
                               "bins": bstats}
        log(f"  bins: " + " | ".join(f"b{b}:rows={bstats[b]['fit_rows']},pos={bstats[b]['fit_pos']},"
                                     f"cand={bstats[b]['n_cand_terms']},"
                                     f"br={bstats[b]['base_rate'] if bstats[b]['base_rate'] is not None else -1:.4f}"
                                     for b in range(N_BINS)))

    # ---- apply calibrations to blind predictions, write arm dirs (vectorized) -----------
    for cat in ("lk", "pk"):
        cs = calib_store[cat]
        P, G, S = [], [], []
        for line in open(DEPLOYED_TSV[cat]):
            p, g, s = line.rstrip("\n").split("\t")
            P.append(p); G.append(g); S.append(float(s))
        S = np.asarray(S, dtype=np.float64)
        TI = np.array([go2idx(g) if g in BP_GO else -1 for g in G])
        is_bp = TI >= 0
        RB = np.array([cs["bin_of_idx"].get(int(t), 1) if t >= 0 else -1 for t in TI])
        # plainIA: single isotonic over all BP rows
        plain = S.copy()
        plain[is_bp] = cs["ir_all"].predict(S[is_bp])
        # freqpart: per-bin isotonic (fallback ir_all if a bin failed to fit)
        freqp = S.copy()
        for bnum in range(N_BINS):
            sel = is_bp & (RB == bnum)
            if not sel.any():
                continue
            ir = cs["iso_bins"].get(bnum) or cs["ir_all"]
            freqp[sel] = ir.predict(S[sel])
        arms = {"deployed": S, "plainIA": plain, "freqpart": freqp}
        for arm, vals in arms.items():
            d = f"{W}/pred_{cat}_{arm}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/{cat}.tsv", "w") as fh:
                for p, g, v in zip(P, G, vals):
                    fh.write(f"{p}\t{g}\t{v:.6f}\n")
        log(f"  {cat}: wrote deployed/plainIA/freqpart dirs")

    json.dump(result, open(f"{W}/calib_build.json", "w"), indent=2)
    log("calib_build.json written")


if __name__ == "__main__":
    main()
