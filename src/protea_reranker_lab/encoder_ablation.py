"""Learned-encoder ablation: dense vs PCA vs a learned GO-aligned projection, on the task metric.

A first-class study (Idiom B, like ``universal_runner``) that asks: over a single PLM's mean-pooled
embeddings, how much does a supervised, GO-aligned learned projection beat the raw dense embedding
and an unsupervised PCA projection, measured on the REAL LAFA frame (cafaeval IA-weighted f_micro_w)?

Each arm produces a per-protein representation; the same KNN GO-transfer + cafaeval pipeline scores
all arms identically, so the deltas are clean:

  - ``dense``    : the raw mean-pooled embedding (baseline = the platform's KNN representation).
  - ``pca``      : unsupervised PCA(k) fit on the reference pool (variance, NOT function).
  - ``learned``  : a trained Linear(d -> dict) projection, top-k real, with cosine(z_i, z_j) ~ Lin
                   GO-similarity (supervised, function-aligned). Objective ``cosine-lin`` or
                   ``hard-neg`` (augments with embedding-near / GO-far mined pairs).

Split is leakage-clean and temporal: reference/train = the t0 (v227) annotated pool (GO-transfer
source + encoder train set); query/eval = the official LAFA targets (excluded from the reference).
Reuses the framework cafaeval recipe (PROTEA venv) and band-registry OBO/IA resolution. GPU for the
learned arm. MLflow-tracked.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

# torch and psycopg2 are imported lazily inside the functions that use them, so the module stays
# importable (autodoc, unit-test collection, the numpy-only helpers) in envs without the DB driver
# or the heavy DL stack installed. This matches the sibling train/universal_train modules.
from protea_reranker_lab.band_registry_bridge import resolve_band_artifacts
from protea_reranker_lab.native_boosters_mlflow import MlflowLogger
from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise, propagate
from protea_reranker_lab.universal_runner import (
    _DEFAULT_PROTEA_PYTHON,
    _run_cafaeval,
    ASPECT_TO_NS,
)

if TYPE_CHECKING:  # annotations only; not evaluated at runtime (from __future__ import annotations)
    import torch.nn as nn

log = logging.getLogger("encoder-ablation")

_ASP_CODE = {"F": "mfo", "P": "bpo", "C": "cco"}  # GT aspect column -> cafaeval cell aspect
_NS_TO_ASP = {v: k for k, v in _ASP_CODE.items()}  # cell aspect -> GT aspect column (PK-known)


# --------------------------------------------------------------------------- spec
@dataclass
class ArmSpec:
    """One representation arm. ``kind`` in {dense, pca, learned}."""
    name: str
    kind: str
    pca_dim: int = 256
    dict_dim: int = 2048
    top_k: int = 128
    objective: str = "cosine-lin"  # learned only: "cosine-lin" | "hard-neg"


def _default_arms() -> list[ArmSpec]:
    return [
        ArmSpec(name="dense", kind="dense"),
        ArmSpec(name="pca256", kind="pca", pca_dim=256),
        ArmSpec(name="learned-k128-coslin", kind="learned", dict_dim=2048, top_k=128,
                objective="cosine-lin"),
        ArmSpec(name="learned-k128-hardneg", kind="learned", dict_dim=2048, top_k=128,
                objective="hard-neg"),
    ]


@dataclass
class EncoderAblationSpec:
    """Configuration for one encoder-ablation run (single PLM)."""
    name: str = "encoder-ablation-esm2-150m"
    dsn: str = "host=localhost dbname=protea user=protea password=protea"
    embedding_config_id: str = "500a0c59-be09-424d-9d51-b7997629c95a"  # esm2_150m, 640d, smallest
    annotation_set_id: str = "c905dffa-a5ce-430b-b17b-503e88666adb"     # GOA v227, t0
    band: str = "v227"
    gt_dir: Path = field(default_factory=lambda: Path(
        "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026"))
    ref_n: int = 60000
    knn: int = 30
    epochs: int = 150
    train_pairs: int = 300_000
    seed: int = 42
    arms: list[ArmSpec] = field(default_factory=_default_arms)
    # Official LAFA harness: score with the exact run_cafa_evaluation recipe
    # (toi_file + PK-known exclusion + th_step=0.01 + max_terms=None) instead of
    # the plain cafaeval. Makes the f_micro_w directly comparable to the 0.3745
    # champion's KNN baseline (still an isolated KNN arm, no reranker).
    official_harness: bool = False
    toi_path: Path | None = None
    pk_known_path: Path | None = None
    out_dir: Path | None = None
    protea_python: Path = field(default_factory=lambda: Path(_DEFAULT_PROTEA_PYTHON))
    mlflow_experiment: str = "encoder-ablation"

    def spec_hash(self) -> str:
        import hashlib
        payload = json.dumps({
            "emb": self.embedding_config_id, "ann": self.annotation_set_id, "band": self.band,
            "ref_n": self.ref_n, "knn": self.knn, "epochs": self.epochs,
            "train_pairs": self.train_pairs, "seed": self.seed,
            "arms": [a.__dict__ for a in self.arms],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- numeric utils
def l2n(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X: np.ndarray, k: int) -> np.ndarray:
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


# --------------------------------------------------------------------------- data layer
def _parse_halfvec(text: str) -> np.ndarray:
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def pull_mean(cur, accs: list[str], cfg: str) -> dict[str, np.ndarray]:
    """accession -> mean-pooled embedding over its chunks, for the given config."""
    cur.execute(
        """SELECT p.accession, se.embedding::text
           FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
           WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)
           ORDER BY p.accession, se.chunk_index_s""", (cfg, list(accs)))
    chunks: dict[str, list] = {}
    for acc, emb in cur.fetchall():
        chunks.setdefault(acc, []).append(_parse_halfvec(emb))
    return {a: np.vstack(v).mean(0).astype(np.float32) for a, v in chunks.items()}


def load_gt(gt_dir: Path) -> dict[tuple[str, str], dict]:
    """Official 7401 GT per cell. cell-key (cat, aspect) -> {proteins:set, pairs:set((acc, go))}."""
    cells: dict[tuple[str, str], dict] = {}
    for cat in ("NK", "LK", "PK"):
        for ln in open(gt_dir / f"groundtruth_{cat}.tsv"):
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] == "EntryID":
                continue
            acc, go, asp = parts[0], parts[1], parts[2]
            aspect = _ASP_CODE.get(asp)
            if aspect is None:
                continue
            key = (cat.lower(), aspect)
            d = cells.setdefault(key, {"proteins": set(), "pairs": set()})
            d["proteins"].add(acc)
            d["pairs"].add((acc, go))
    return cells


# --------------------------------------------------------------------------- representations
def _pca(R: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Unsupervised PCA(k) fit on the reference pool (transductive), applied to ref + query."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(k, R.shape[1], R.shape[0]), random_state=0).fit(R)
    return p.transform(R).astype(np.float32), p.transform(Q).astype(np.float32)


def fit_encoder(R: np.ndarray, closures: list[frozenset[str]], dag: GoDag, arm: ArmSpec,
                spec: EncoderAblationSpec) -> nn.Linear:
    """Train the GO-aligned Linear(d->dict) projection on the reference pool; return the model."""
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(spec.seed)
    d = R.shape[1]
    Rt = torch.tensor(l2n(R), device=dev)
    ic = information_content(closures, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]

    pairs = sample_pairs(len(closures), spec.train_pairs, rng)
    if arm.objective == "hard-neg":
        # mine embedding-near pairs (hard: close in PLM space, target = their true Lin, often low)
        Rn = l2n(R)
        anchors = rng.choice(len(closures), size=min(2000, len(closures)), replace=False)
        S = Rn[anchors] @ Rn.T
        np.put_along_axis(S, anchors[:, None], -1.0, axis=1)
        nbr = np.argpartition(-S, spec.knn, axis=1)[:, :spec.knn]
        for a_i, anchor in enumerate(anchors):
            for j in nbr[a_i]:
                pairs.append((int(anchor), int(j)) if anchor < j else (int(j), int(anchor)))

    y = torch.tensor(np.asarray(lin_pairwise(closures, ic, pairs, bic), dtype=np.float32), device=dev)
    ti = torch.tensor([p[0] for p in pairs], device=dev)
    tj = torch.tensor([p[1] for p in pairs], device=dev)
    enc = nn.Linear(d, arm.dict_dim).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    bs = 32768
    np_ = len(pairs)
    for e in range(spec.epochs):
        z = enc(Rt)
        opt.zero_grad()
        loss = torch.zeros((), device=dev)
        for b in range(0, np_, bs):
            sl = slice(b, b + bs)
            zi, zj = z[ti[sl]], z[tj[sl]]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = loss + ((cos - y[sl]) ** 2).sum() / np_
        loss.backward()
        opt.step()
        if e % 30 == 0:
            log.info("  [%s] enc epoch %3d loss=%.4f", arm.name, e, float(loss))
    return enc


def apply_encoder(enc: nn.Linear, X: np.ndarray, top_k: int) -> np.ndarray:
    """Project L2-normalised embeddings through the encoder and keep the top-k real code."""
    import torch

    dev = next(enc.parameters()).device
    with torch.no_grad():
        Z = enc(torch.tensor(l2n(X), device=dev)).cpu().numpy().astype(np.float32)
    return topk_real(Z, top_k)


def _train_encoder(
    R: np.ndarray, Q: np.ndarray, closures: list[frozenset[str]], dag: GoDag, arm: ArmSpec,
    spec: EncoderAblationSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Train + apply: return top-k real codes for ref + query (the ablation's learned arm)."""
    enc = fit_encoder(R, closures, dag, arm, spec)
    return apply_encoder(enc, R, arm.top_k), apply_encoder(enc, Q, arm.top_k)


def _build_arm(arm: ArmSpec, R: np.ndarray, Q: np.ndarray, closures: list[frozenset[str]], dag: GoDag,
               spec: EncoderAblationSpec) -> tuple[np.ndarray, np.ndarray]:
    if arm.kind == "dense":
        return R, Q
    if arm.kind == "pca":
        return _pca(R, Q, arm.pca_dim)
    if arm.kind == "learned":
        return _train_encoder(R, Q, closures, dag, arm, spec)
    raise ValueError(f"unknown arm kind: {arm.kind}")


# --------------------------------------------------------------------------- GO transfer
def knn_transfer(Q: np.ndarray, R: np.ndarray, ref_closures: list[frozenset[str]], terms_ix: dict[str, int],
                 knn: int, batch: int = 1000) -> sp.csr_matrix:
    """Each query: cosine top-knn in the reference pool, similarity-weighted vote over GO closures."""
    Rn = l2n(R)
    Qn = l2n(Q)
    nq, nt = Qn.shape[0], len(terms_ix)
    rr, rc = [], []
    for i, cl in enumerate(ref_closures):
        for t in cl:
            rr.append(i)
            rc.append(terms_ix[t])
    refT = sp.csr_matrix((np.ones(len(rr), np.float32), (rr, rc)), shape=(R.shape[0], nt))
    out = sp.lil_matrix((nq, nt), dtype=np.float32)
    for b in range(0, nq, batch):
        sl = slice(b, min(b + batch, nq))
        S = Qn[sl] @ Rn.T
        nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
        bq = S.shape[0]
        w = np.take_along_axis(S, nbr, axis=1)
        w[w < 0] = 0
        rows = np.repeat(np.arange(bq), knn)
        Sk = sp.csr_matrix((w.ravel(), (rows, nbr.ravel())), shape=(bq, R.shape[0]), dtype=np.float32)
        out[sl] = (Sk @ refT)
    return out.tocsr()


def _write_cell_tsvs(scores: sp.csr_matrix, query_ix: dict[str, int], terms: list[str],
                     cells: dict, work_dir: Path, max_terms: int = 500) -> None:
    """Write cafaeval pred dir (model.tsv) + gt.tsv per cell, restricted to the cell's proteins."""
    inv = {i: a for a, i in query_ix.items()}
    sc = scores.tocoo()
    per: dict[int, list] = {}
    for r, c, v in zip(sc.row, sc.col, sc.data):
        if v > 0:
            per.setdefault(r, []).append((float(v), c))
    for (cat, aspect), meta in cells.items():
        cell = f"{cat}-{aspect}"
        cell_dir = work_dir / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        prots = meta["proteins"]
        with open(cell_dir / "model.tsv", "w") as fp:
            for r, lst in per.items():
                acc = inv.get(r)
                if acc is None or acc not in prots:
                    continue
                for v, c in sorted(lst, reverse=True)[:max_terms]:
                    fp.write(f"{acc}\t{terms[c]}\t{v:.6f}\n")
        with open(cell_dir / "gt.tsv", "w") as g:
            for acc, go in meta["pairs"]:
                g.write(f"{acc}\t{go}\n")


# ----------------------------------------------------------------- official LAFA harness
_CAFAEVAL_DRIVER_OFFICIAL = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
signal.signal(signal.SIGINT, signal.SIG_DFL)
df, dfs_best = cafa_eval(
    "{obo}", "{pred_dir}", "{gt}",
    ia="{ia}", prop="fill", norm="cafa", no_orphans=True,
    toi_file="{toi}", exclude={exclude}, max_terms=None, th_step=0.01,
    n_cpu=1, weighted_only=False,
)
out = {{}}
for kind, df_best in dfs_best.items():
    out[kind] = df_best.reset_index().to_dict(orient="records")
with open("{out_json}", "w") as f:
    json.dump(out, f, indent=2, default=str)
'''


def _load_pk_known(pk_known_path: Path) -> dict[str, set[tuple[str, str]]]:
    """aspect-code (F/P/C) -> {(protein, term)} of already-known PK annotations to exclude."""
    out: dict[str, set[tuple[str, str]]] = {}
    for ln in open(pk_known_path):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[0] == "EntryID":
            continue
        out.setdefault(parts[2], set()).add((parts[0], parts[1]))
    return out


def _run_cafaeval_official(cell: str, work_dir: Path, obo_path: Path, ia_path: Path,
                           exclude_path: Path | None, spec: EncoderAblationSpec) -> dict:
    """cafaeval with the exact LAFA recipe (toi_file + PK-known exclude + th_step=0.01)."""
    import subprocess
    cell_dir = work_dir / cell
    out_json = cell_dir / "cafaeval_official.json"
    excl = "None" if exclude_path is None else repr(str(exclude_path))
    driver = _CAFAEVAL_DRIVER_OFFICIAL.format(
        obo=str(obo_path), pred_dir=str(cell_dir), gt=str(cell_dir / "gt.tsv"),
        ia=str(ia_path), toi=str(spec.toi_path), exclude=excl, out_json=str(out_json))
    try:
        subprocess.run([str(spec.protea_python), "-c", driver], timeout=1800, check=True,
                       capture_output=True, text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)[:200]}
    if not out_json.exists():
        return {"error": "cafaeval produced no output"}
    raw = json.loads(out_json.read_text())
    ns = ASPECT_TO_NS.get(cell.split("-", 1)[1])
    for row in raw.get("f_micro_w", []):
        if row.get("ns") == ns and row.get("f_micro_w") is not None:
            return {"f_micro_w": float(row["f_micro_w"])}
    return {"f_micro_w": None}


# --------------------------------------------------------------------------- runner
def _load_data(spec: EncoderAblationSpec, dag: GoDag, queries: list[str], rng):
    """Pull the t0 reference pool (embeddings + GO closures) and the query embeddings (read-only)."""
    import psycopg2

    conn = psycopg2.connect(spec.dsn)
    cur = conn.cursor()
    qset = set(queries)
    cur.execute(
        """SELECT DISTINCT protein_accession FROM protein_go_annotation
           WHERE annotation_set_id = %s AND (hashtextextended(protein_accession, 42) %% %s) = 0""",
        (spec.annotation_set_id, max(2, 556000 // (spec.ref_n * 2))))
    ref_accs = [a for (a,) in cur.fetchall() if a not in qset]
    rng.shuffle(ref_accs)
    ref_accs = ref_accs[:spec.ref_n]
    ref_emb = pull_mean(cur, ref_accs, spec.embedding_config_id)
    q_emb = pull_mean(cur, queries, spec.embedding_config_id)
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id FROM protein_go_annotation pga
           JOIN go_term gt ON gt.id = pga.go_term_id
           WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)""",
        (spec.annotation_set_id, list(ref_emb.keys())))
    leaves: dict[str, list] = {}
    for a, go in cur.fetchall():
        leaves.setdefault(a, []).append(go)
    cur.close()
    conn.close()

    keep = [a for a in ref_emb if a in leaves and propagate(leaves[a], dag)]
    ref_clo = [propagate(leaves[a], dag) for a in keep]
    R = np.vstack([ref_emb[a] for a in keep]).astype(np.float32)
    q_accs = [a for a in queries if a in q_emb]
    Q = np.vstack([q_emb[a] for a in q_accs]).astype(np.float32)
    return R, Q, ref_clo, q_accs


def _log_mlflow(spec: EncoderAblationSpec, out_dir: Path, results: dict, ref_n: int) -> None:
    logger = MlflowLogger.maybe_create(
        run_name=spec.name, experiment=os.environ.get("MLFLOW_EXPERIMENT", spec.mlflow_experiment))
    if logger is None:
        return
    with logger:
        logger.log_run_params({
            "embedding_config_id": spec.embedding_config_id, "band": spec.band,
            "reference_n": ref_n, "knn": spec.knn, "epochs": spec.epochs,
            "arms": ",".join(a.name for a in spec.arms),
        }, tags={"study": "encoder-ablation"})
        for name, r in results.items():
            for cell, v in r["cells"].items():
                logger._safe(lambda n=name, c=cell, val=v: logger._mlflow.log_metric(
                    f"{n}__{c}".replace("-", "_"), val))
            if r["nklk_mean"] is not None:
                logger._safe(lambda n=name, val=r["nklk_mean"]: logger._mlflow.log_metric(
                    f"{n}__nklk_mean".replace("-", "_"), val))
        logger.log_summary_artifact(out_dir / "run.json")


def _eval_arm_cells(work: Path, cells: dict, obo_path: Path, ia_path: Path,
                    spec: EncoderAblationSpec, pk_known: dict) -> dict[str, float]:
    """Score every cell, choosing the official LAFA harness (toi + PK-known exclude) or plain."""
    cell_fw: dict[str, float] = {}
    for (cat, aspect) in cells:
        cell = f"{cat}-{aspect}"
        if spec.official_harness and spec.toi_path is not None:
            excl = None
            if cat == "pk" and pk_known:
                excl = work / cell / "exclude.tsv"
                with open(excl, "w") as f:
                    for acc, term in pk_known.get(_NS_TO_ASP[aspect], set()):
                        f.write(f"{acc}\t{term}\n")
            m = _run_cafaeval_official(cell, work, obo_path, ia_path, excl, spec)
        else:
            m = _run_cafaeval(cell, work, obo_path, ia_path, spec.protea_python)
        fw = m.get("f_micro_w") if isinstance(m, dict) else None
        if isinstance(fw, (int, float)):
            cell_fw[cell] = float(fw)
        log.info("  %-7s f_micro_w=%s", cell, f"{fw:.4f}" if isinstance(fw, (int, float)) else "NA")
    return cell_fw


def train_and_save_encoder(spec: EncoderAblationSpec, arm: ArmSpec, out_path: Path) -> dict:
    """Train ONE production encoder on the reference pool and persist its weights + meta.

    The artifact (torch ``{state_dict, meta}``) is what a downstream apply step (e.g. a PROTEA
    operation) loads to project any protein's mean-pooled embedding into the learned code:
    ``topk_real(enc(l2n(x)), top_k)``. Meta carries everything needed to reconstruct + apply it.
    """
    import torch

    obo_path, _ = resolve_band_artifacts(spec.band)
    cells = load_gt(spec.gt_dir)
    queries = sorted({a for d in cells.values() for a in d["proteins"]})
    dag = GoDag.from_obo(obo_path)
    R, _Q, ref_clo, _q = _load_data(spec, dag, queries, np.random.default_rng(spec.seed))
    log.info("training production encoder (%s) on %d reference proteins (dim=%d)",
             arm.name, len(ref_clo), R.shape[1])
    enc = fit_encoder(R, ref_clo, dag, arm, spec)
    meta = {
        "in_dim": int(R.shape[1]), "dict_dim": arm.dict_dim, "top_k": arm.top_k,
        "objective": arm.objective, "source_embedding_config_id": spec.embedding_config_id,
        "l2_normalize_input": True, "band": spec.band, "reference_n": len(ref_clo),
        "seed": spec.seed,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in enc.state_dict().items()}, "meta": meta},
               out_path)
    log.info("saved encoder -> %s | meta=%s", out_path, meta)
    return meta


def run_encoder_ablation(spec: EncoderAblationSpec) -> dict:
    """Run the ablation: pull data, build each arm, KNN-transfer, cafaeval, collect deltas."""
    out_dir = Path(spec.out_dir or (Path("runs") / "encoder_ablation" / spec.spec_hash()))
    out_dir.mkdir(parents=True, exist_ok=True)
    obo_path, ia_path = resolve_band_artifacts(spec.band)
    log.info("spec=%s hash=%s | OBO=%s IA=%s", spec.name, spec.spec_hash(), obo_path.name, ia_path.name)

    cells = load_gt(spec.gt_dir)
    queries = sorted({a for d in cells.values() for a in d["proteins"]})
    dag = GoDag.from_obo(obo_path)
    R, Q, ref_clo, q_accs = _load_data(spec, dag, queries, np.random.default_rng(spec.seed))
    log.info("reference=%d (with closures) | queries=%d/%d | dim=%d",
             len(ref_clo), len(q_accs), len(queries), R.shape[1])
    terms = sorted({t for c in ref_clo for t in c})
    tix = {t: i for i, t in enumerate(terms)}
    query_ix = {a: i for i, a in enumerate(q_accs)}

    pk_known = (_load_pk_known(spec.pk_known_path)
                if (spec.official_harness and spec.pk_known_path) else {})
    if spec.official_harness:
        log.info("OFFICIAL LAFA harness: toi=%s + PK-known exclusion", str(spec.toi_path))

    results: dict[str, dict] = {}
    for arm in spec.arms:
        log.info("=== arm=%s (%s) ===", arm.name, arm.kind)
        Rx, Qx = _build_arm(arm, R, Q, ref_clo, dag, spec)
        scores = knn_transfer(Qx, Rx, ref_clo, tix, spec.knn)
        work = out_dir / "cafaeval" / arm.name
        _write_cell_tsvs(scores, query_ix, terms, cells, work)
        cell_fw = _eval_arm_cells(work, cells, obo_path, ia_path, spec, pk_known)
        nklk = [v for c, v in cell_fw.items() if c.split("-")[0] in ("nk", "lk")]
        pk = [v for c, v in cell_fw.items() if c.startswith("pk")]
        results[arm.name] = {
            "cells": cell_fw,
            "nklk_mean": float(np.mean(nklk)) if nklk else None,
            "pk_mean": float(np.mean(pk)) if pk else None,
        }

    base = results.get("dense", {}).get("nklk_mean")
    log.info("=== SUMMARY (f_micro_w on the official %s frame) ===", spec.band)
    for name, r in results.items():
        d = (r["nklk_mean"] - base) if (base is not None and r["nklk_mean"] is not None) else None
        log.info("  %-22s NK+LK=%s PK=%s %s", name,
                 f"{r['nklk_mean']:.4f}" if r["nklk_mean"] is not None else "NA",
                 f"{r['pk_mean']:.4f}" if r["pk_mean"] is not None else "NA",
                 f"({d:+.4f} vs dense)" if d is not None else "")

    report = {
        "name": spec.name, "spec_hash": spec.spec_hash(), "status": "ok",
        "embedding_config_id": spec.embedding_config_id, "band": spec.band,
        "reference_n": len(ref_clo), "queries": len(q_accs), "dim": int(R.shape[1]),
        "results": results,
    }
    (out_dir / "run.json").write_text(json.dumps(report, indent=2, default=str))
    _log_mlflow(spec, out_dir, results, len(ref_clo))
    log.info("run.json -> %s", out_dir / "run.json")
    return report
