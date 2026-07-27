"""B: per-cut temporal GO sparse codes (aligned basis).

The two-tower head is FROZEN and was trained against the v227 GO codes, whose
co-annotation block lives in the v227 TruncatedSVD basis. Per-cut codes must
stay in THAT SAME basis or the frozen head is mis-aligned. So the SVD basis is
part of the frozen method: fit it ONCE on v227, then for every cut build that
cut's own PPMI over the fixed v227 term-column index and TRANSFORM (not refit)
it through the frozen basis. Text block is t0-independent and identical across
cuts. Output one go_sparse_codes_v{N}.npz per cut into per_cut/.

Consistency gate: the v227 artifact rebuilt here must match the existing serve
go_sparse_codes.npz (which the head trained on); if it does, all other cuts are
in the same basis by construction.

No PROTEA import; direct psycopg2 read-only.
"""
import os, sys, time, tempfile, numpy as np, pandas as pd, psycopg2
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD

R = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"
OUT = "/home/frapercan/Thesis2/storage/two_tower_sparse/per_cut"
TXT_KWTA, COA_KWTA, SVD_DIM = 96, 96, 256
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"

# cut version -> annotation_set_id (t0 sets needed for the 14 train pairs + eval t0)
CUTS = {
    160: "a8e2ffd4-721c-493f-882b-db6b92cb98bd",
    165: "d09b2b14-0b27-47d0-afd9-8c9e5c3c3052",
    170: "f720c3c4-abc0-49f7-9d3c-bb917a77089c",
    175: "eb33207a-d876-420e-be67-8f0bd2748909",
    180: "2a5f3325-e4e8-4f6f-8f0c-82b1f6ca755f",
    185: "55999285-e523-4388-999d-fe934cc09994",
    190: "120edbcf-467a-4a76-8c36-a86cc51bed3d",
    195: "5db5ea46-4a37-4f37-a282-5a97e98207c6",
    200: "9493bc57-90d6-4929-ac02-f897af0c04e6",
    205: "3c51a9c8-d7a2-40c5-86a3-30d659542e73",
    211: "22cb2901-09ca-49fa-8ec9-271d3feda62e",
    215: "b66eb37f-9e9d-4ee5-a6cd-c5379abbae30",
    220: "1559d9f7-195d-4892-af16-8b58f7fc9942",
    225: "38280963-c8cd-4a2b-95c9-fa682ecc232d",
    227: "c905dffa-a5ce-430b-b17b-503e88666adb",
}


def kwta_rows(X, k):
    Xk = np.zeros_like(X)
    idx = np.argpartition(np.abs(X), -k, axis=1)[:, -k:]
    rows = np.arange(X.shape[0])[:, None]
    Xk[rows, idx] = X[rows, idx]
    n = np.linalg.norm(Xk, axis=1, keepdims=True); n[n == 0] = 1
    return (Xk / n).astype(np.float32)


def load_cooc(aset):
    """COPY (k, c, n) co-annotation rows for one annotation set."""
    conn = psycopg2.connect("host=localhost dbname=protea user=protea password=protea")
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        path = f.name
        conn.cursor().copy_expert(
            f"COPY (SELECT known_go_id, candidate_go_id, cooccurrence_count "
            f"FROM term_cooccurrence WHERE annotation_set_id='{aset}') TO STDOUT WITH CSV", f)
    conn.close()
    df = pd.read_csv(path, header=None, names=["k", "c", "n"])
    os.unlink(path)
    return df


def ppmi_over_index(df, ti, N):
    """Symmetric PPMI sparse matrix on a FIXED term index ti (size N)."""
    df = df[df.k.isin(ti) & df.c.isin(ti)]
    ri = df.k.map(ti).to_numpy(); ci = df.c.map(ti).to_numpy(); v = df.n.to_numpy(np.float64)
    M = coo_matrix((v, (ri, ci)), shape=(N, N)).tocsr()
    M = M + M.T
    M = M.tocoo(); tot = M.data.sum()
    rs = np.asarray(csr_matrix((M.data, (M.row, M.col)), shape=(N, N)).sum(1)).ravel()
    rs[rs == 0] = 1.0
    pmi = np.log((M.data * tot) / (rs[M.row] * rs[M.col]) + 1e-12)
    pmi[pmi < 0] = 0
    return csr_matrix((pmi, (M.row, M.col)), shape=(N, N))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d = np.load(f"{R}/go_text_emb.npz", allow_pickle=True)
    txt_ids = list(d["go_ids"]); T = d["emb"].astype(np.float32)
    T = T - T.mean(0, keepdims=True)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    Tk = kwta_rows(T, TXT_KWTA)
    print(f"[{time.time()-t0:.0f}s] text {Tk.shape}", flush=True)

    # --- FROZEN BASIS: fit SVD once on v227, fixing the term-column index ---
    # DETERMINISTIC term order (sorted) so the basis is reproducible forever, and
    # SAVE the SVD components so per-cut transforms (and any future cut) reuse the
    # exact same basis without refitting. This closes the reproducibility gap: the
    # original serve basis used pd.unique (COPY row order) and was never saved.
    df227 = load_cooc(V227)
    terms = pd.Index(sorted(pd.unique(pd.concat([df227.k, df227.c]))))
    ti = {g: i for i, g in enumerate(terms)}; N = len(terms)
    P227 = ppmi_over_index(df227, ti, N)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=0).fit(P227)
    np.savez(f"{OUT}/svd_basis.npz", components=svd.components_.astype(np.float32),
             terms=np.array(terms), svd_dim=SVD_DIM)
    print(f"[{time.time()-t0:.0f}s] frozen SVD basis fit on v227 terms={N} "
          f"(deterministic, saved svd_basis.npz)", flush=True)

    def codes_for(df):
        CA = svd.transform(ppmi_over_index(df, ti, N)).astype(np.float32)
        CA = CA / (np.linalg.norm(CA, axis=1, keepdims=True) + 1e-8)
        coann_of = {terms[i]: CA[i] for i in range(N)}
        CAfull = np.zeros((len(txt_ids), SVD_DIM), np.float32); hit = 0
        for i, g in enumerate(txt_ids):
            u = coann_of.get(g)
            if u is not None:
                CAfull[i] = u; hit += 1
        CAk = kwta_rows(CAfull, COA_KWTA)
        return np.hstack([Tk, CAk]).astype(np.float32), hit

    # consistency gate on v227 vs the serve artifact the head trained on
    fused227, hit227 = codes_for(df227)
    serve = np.load("/home/frapercan/Thesis2/storage/two_tower_sparse/go_sparse_codes.npz", allow_pickle=True)
    sv = serve["codes"].astype(np.float32)
    if sv.shape == fused227.shape:
        diff = float(np.abs(sv - fused227).max())
        rowcos = float(np.mean(np.sum(sv * fused227, 1) / (np.linalg.norm(sv,axis=1)*np.linalg.norm(fused227,axis=1)+1e-9)))
        print(f"[gate] v227 rebuild vs serve: max|d|={diff:.4f} mean_rowcos={rowcos:.4f} cov={hit227}/{len(txt_ids)}", flush=True)
    else:
        print(f"[gate] WARN shape mismatch serve {sv.shape} vs rebuild {fused227.shape}", flush=True)

    only = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(CUTS)
    for v in only:
        aset = CUTS[v]
        df = df227 if aset == V227 else load_cooc(aset)
        fused, hit = codes_for(df)
        outp = f"{OUT}/go_sparse_codes_v{v}.npz"
        np.savez(outp, go_ids=np.array(txt_ids), codes=fused,
                 text_kwta=TXT_KWTA, coann_kwta=COA_KWTA)
        print(f"[{time.time()-t0:.0f}s] v{v} -> {outp} coann_cov={hit}/{len(txt_ids)}", flush=True)
    print(f"[{time.time()-t0:.0f}s] DONE {len(only)} cuts", flush=True)


if __name__ == "__main__":
    main()
