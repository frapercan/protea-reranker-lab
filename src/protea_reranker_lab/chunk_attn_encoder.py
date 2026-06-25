"""Train + save a chunk-level ATTENTION-pool learned encoder for the platform.

The plain learned encoder (``encoder_ablation`` / ``apply_learned_encoder``) is
``topk_real(Linear(mean_pool(chunks)))`` -- a MAGNITUDE-agnostic mean over a protein's
per-chunk vectors followed by one Linear projection into a GO-aligned top-k code.

The "length winner" of the chunk proxy is an ATTENTION pool over the per-chunk vectors:
a small additive single/multi-head scorer learns *which chunks matter* (domains, active
sites) and weights them before the Linear, instead of averaging them uniformly. This
module trains that encoder against the same GO-contrastive (Lin similarity, with mined
embedding-near / GO-far hard negatives) objective used by ``encoder_ablation`` and saves
a torch artifact that ``apply_learned_encoder`` (pooling="attention") can apply in-platform
over the PER-CHUNK rows of a source ``EmbeddingConfig``.

Artifact layout (what the op consumes)::

    {"state_dict": <AttnPoolChunkEncoder state>,
     "meta": {"in_dim", "dict_dim", "top_k", "att_dim", "heads",
              "pooling": "attention", "objective", "source_embedding_config_id",
              "l2_normalize_chunks", "band", "reference_n", "seed"}}

Read-only DB, GPU, minibatched (the OOM-safe pattern). Leakage-clean: trains only on a
t0 reference pool + the matching t0 OBO snapshot.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from protea_reranker_lab.band_registry_bridge import resolve_band_artifacts
from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise, propagate

if TYPE_CHECKING:  # heavy / optional deps stay out of import time (lab CI lazy-import rule)
    import torch
    import torch.nn as nn

log = logging.getLogger("chunk-attn-encoder")


# --------------------------------------------------------------------------- spec
@dataclass
class ChunkAttnEncoderSpec:
    """Configuration for one chunk-attention encoder training run."""

    dsn: str = "host=localhost dbname=protea user=protea password=protea"
    # ankh-base per-chunk config (6542db1e), 768d, chunked + materialised.
    source_embedding_config_id: str = "6542db1e-202a-4769-b933-2e0f85aa81e6"
    annotation_set_id: str = "c905dffa-a5ce-430b-b17b-503e88666adb"  # GOA v227, t0
    band: str = "v227"
    dict_dim: int = 2048
    top_k: int = 128
    att_dim: int = 128
    heads: int = 1  # single-head gated (light-attention-like); 4 = PMA-like multi-head
    cap_chunks: int = 16  # pad/truncate per-protein chunk count (ankh-base maxes at 10)
    reference_n: int = 100_000
    epochs: int = 150
    train_pairs: int = 300_000
    knn_hardneg: int = 30
    lr: float = 1e-3
    seed: int = 42
    objective: str = "hard-neg"  # "hard-neg" | "cosine-lin"
    out_path: Path = field(
        default_factory=lambda: Path(
            "/home/frapercan/Thesis2/storage/learned_encoders/ankh_base_chunk_attnpool.pt"
        )
    )


# --------------------------------------------------------------------------- encoder
def build_attn_encoder(in_dim: int, dict_dim: int, att_dim: int, heads: int) -> "nn.Module":
    """Construct the chunk-attention-pool encoder (additive attention over chunks + Linear)."""
    import torch
    import torch.nn as nn

    class AttnPoolChunkEncoder(nn.Module):
        """Additive multi-head attention pool over per-chunk vectors, then a Linear projection.

        ``forward(M, mask)`` with ``M``: (B, cap, in_dim) padded per-chunk vectors and
        ``mask``: (B, cap) bool (True = real chunk). Produces (B, dict_dim) codes.
        """

        def __init__(self) -> None:
            super().__init__()
            self.heads = heads
            self.W = nn.Linear(in_dim, att_dim)
            self.v = nn.Linear(att_dim, heads, bias=False)
            self.lin = nn.Linear(in_dim * heads, dict_dim)

        def forward(self, M: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            s = self.v(torch.tanh(self.W(M)))  # (B, cap, heads) content scores
            s = s.masked_fill(~mask.unsqueeze(-1), -1e9)
            a = torch.softmax(s, dim=1)  # attention over chunks, per head
            pooled = torch.einsum("bch,bcd->bhd", a, M)  # (B, heads, in_dim)
            return self.lin(pooled.reshape(M.shape[0], -1))

    return AttnPoolChunkEncoder()


def topk_real(X: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest-magnitude entries per row (zero the rest) -- the sparse code."""
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def _sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
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


def _l2n(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return (X / nrm).astype(np.float32)


# --------------------------------------------------------------------------- data
def _parse_halfvec(text: str) -> np.ndarray:
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)


def pull_chunks(cur, accs: list[str], cfg: str) -> dict[str, list[np.ndarray]]:
    """accession -> ordered list of its per-chunk embedding vectors (by chunk_index_s)."""
    cur.execute(
        """SELECT p.accession, se.embedding::text
           FROM protein p JOIN sequence_embedding se ON se.sequence_id = p.sequence_id
           WHERE se.embedding_config_id = %s AND p.accession = ANY(%s)
           ORDER BY p.accession, se.chunk_index_s""",
        (cfg, list(accs)),
    )
    chunks: dict[str, list[np.ndarray]] = {}
    for acc, emb in cur.fetchall():
        chunks.setdefault(acc, []).append(_parse_halfvec(emb))
    return chunks


def _load_reference(spec: ChunkAttnEncoderSpec, dag: GoDag, rng: np.random.Generator):
    """Pull the t0 reference pool: per-chunk vectors + propagated GO closures (read-only)."""
    import psycopg2

    conn = psycopg2.connect(spec.dsn)
    cur = conn.cursor()
    cur.execute(
        """SELECT DISTINCT protein_accession FROM protein_go_annotation
           WHERE annotation_set_id = %s AND (hashtextextended(protein_accession, 42) %% %s) = 0""",
        (spec.annotation_set_id, max(2, 556000 // (spec.reference_n * 2))),
    )
    ref_accs = [a for (a,) in cur.fetchall()]
    rng.shuffle(ref_accs)
    ref_accs = ref_accs[: spec.reference_n]
    chunks = pull_chunks(cur, ref_accs, spec.source_embedding_config_id)
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id FROM protein_go_annotation pga
           JOIN go_term gt ON gt.id = pga.go_term_id
           WHERE pga.annotation_set_id = %s AND pga.protein_accession = ANY(%s)""",
        (spec.annotation_set_id, list(chunks.keys())),
    )
    leaves: dict[str, list[str]] = {}
    for a, go in cur.fetchall():
        leaves.setdefault(a, []).append(go)
    cur.close()
    conn.close()

    keep = [a for a in chunks if a in leaves and propagate(leaves[a], dag)]
    closures = [propagate(leaves[a], dag) for a in keep]
    per_chunk = [chunks[a] for a in keep]
    return keep, per_chunk, closures


def _pad_chunks(per_chunk: list[list[np.ndarray]], cap: int, d: int):
    """List-of-chunk-lists -> padded (n, cap, d) float32 + (n, cap) bool mask (truncate to cap)."""
    n = len(per_chunk)
    Xpad = np.zeros((n, cap, d), np.float32)
    mask = np.zeros((n, cap), bool)
    for i, ch in enumerate(per_chunk):
        m = np.vstack(ch[:cap]).astype(np.float32)
        Xpad[i, : m.shape[0]] = m
        mask[i, : m.shape[0]] = True
    return Xpad, mask


# --------------------------------------------------------------------------- train
def _build_pairs(spec, n, per_chunk, closures, ic_bic, rng) -> tuple[list, np.ndarray]:
    """Sample training pairs (+ mined hard negatives for the hard-neg objective) and their targets.

    ``ic_bic`` is the ``(information_content, best_ic_per_protein)`` tuple from the closures.
    """
    ic, bic = ic_bic
    pairs = _sample_pairs(n, spec.train_pairs, rng)
    if spec.objective == "hard-neg":
        # mine embedding-near pairs (hard: close in PLM space, GO-similarity often low) over the
        # cheap mean-pooled dense neighbour graph
        rmean = _l2n(np.vstack([np.vstack(ch).mean(0) for ch in per_chunk]).astype(np.float32))
        anchors = rng.choice(n, size=min(2000, n), replace=False)
        sim = rmean[anchors] @ rmean.T
        np.put_along_axis(sim, anchors[:, None], -1.0, axis=1)
        nbr = np.argpartition(-sim, spec.knn_hardneg, axis=1)[:, : spec.knn_hardneg]
        for a_i, anchor in enumerate(anchors):
            for j in nbr[a_i]:
                pairs.append((int(anchor), int(j)) if anchor < j else (int(j), int(anchor)))
    y = np.asarray(lin_pairwise(closures, ic, pairs, bic), dtype=np.float32)
    return pairs, y


def _fit(spec, enc, opt, padded, pairs_y, dev) -> None:
    """GPU-memory-bounded optimisation: per pair-batch, encode only referenced proteins, backprop.

    ``padded`` is the ``(Xpad, mask)`` tuple; ``pairs_y`` is ``(pairs, target_lin_similarities)``.
    The padded ``(n, cap, d)`` chunk tensor + targets stay on CPU; each pair-batch moves only its
    unique proteins to GPU, so memory is bounded by one pair-batch instead of the whole ``(n, dict)``
    graph (scales to 100k+ proteins on a 12GB card). Gradients accumulate across pair-batches, one
    ``opt.step()`` per epoch (mathematically the full-batch update).
    """
    import torch

    Xpad, mask = padded
    pairs, y = pairs_y
    Xpad_cpu = torch.tensor(Xpad)
    mask_cpu = torch.tensor(mask)
    y_cpu = torch.tensor(y)
    ti = np.asarray([p[0] for p in pairs], dtype=np.int64)
    tj = np.asarray([p[1] for p in pairs], dtype=np.int64)
    pair_bs = 16384
    np_ = len(pairs)
    for e in range(spec.epochs):
        opt.zero_grad()
        epoch_loss = 0.0
        for b in range(0, np_, pair_bs):
            sl = slice(b, b + pair_bs)
            bi, bj = ti[sl], tj[sl]
            uniq, inv = np.unique(np.concatenate([bi, bj]), return_inverse=True)
            half = len(bi)
            z = enc(Xpad_cpu[uniq].to(dev), mask_cpu[uniq].to(dev))  # (n_uniq, dict)
            zi = z[torch.as_tensor(inv[:half], device=dev)]
            zj = z[torch.as_tensor(inv[half:], device=dev)]
            cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
            loss = ((cos - y_cpu[sl].to(dev)) ** 2).sum() / np_
            loss.backward()
            epoch_loss += float(loss.detach())
        opt.step()
        if e % 30 == 0 or e == spec.epochs - 1:
            log.info("  epoch %3d loss=%.4f", e, epoch_loss)


def train_chunk_attn_encoder(spec: ChunkAttnEncoderSpec) -> dict:
    """Train the chunk-attention encoder on the t0 pool and save the artifact + JSON sidecar.

    Returns the meta dict written into the artifact.
    """
    import torch

    obo_path, _ia = resolve_band_artifacts(spec.band)
    dag = GoDag.from_obo(obo_path)
    rng = np.random.default_rng(spec.seed)
    accs, per_chunk, closures = _load_reference(spec, dag, rng)
    n = len(accs)
    if n == 0:
        raise RuntimeError("empty reference pool (check source config / annotation set)")
    d = int(per_chunk[0][0].shape[0])
    log.info("reference=%d proteins | dim=%d | cap_chunks=%d", n, d, spec.cap_chunks)

    Xpad, mask = _pad_chunks(per_chunk, spec.cap_chunks, d)
    ic = information_content(closures, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]
    pairs, y = _build_pairs(spec, n, per_chunk, closures, (ic, bic), rng)
    log.info("training pairs=%d (objective=%s)", len(pairs), spec.objective)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = build_attn_encoder(d, spec.dict_dim, spec.att_dim, spec.heads).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=spec.lr)
    _fit(spec, enc, opt, (Xpad, mask), (pairs, y), dev)

    meta = {
        "in_dim": d,
        "dict_dim": spec.dict_dim,
        "top_k": spec.top_k,
        "att_dim": spec.att_dim,
        "heads": spec.heads,
        "cap_chunks": spec.cap_chunks,
        "pooling": "attention",
        "objective": spec.objective,
        "source_embedding_config_id": spec.source_embedding_config_id,
        "l2_normalize_chunks": False,
        "band": spec.band,
        "reference_n": n,
        "seed": spec.seed,
    }
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": enc.state_dict(), "meta": meta}, spec.out_path)
    sidecar = spec.out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True))
    log.info("saved artifact -> %s (+ sidecar %s)", spec.out_path, sidecar.name)
    return meta


def main() -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [chunk-attn] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Train + save a chunk-attention learned encoder.")
    ap.add_argument("--source-config", default=ChunkAttnEncoderSpec.source_embedding_config_id)
    ap.add_argument("--out", default=str(ChunkAttnEncoderSpec().out_path))
    ap.add_argument("--dict-dim", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--att-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--cap-chunks", type=int, default=16)
    ap.add_argument("--reference-n", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--train-pairs", type=int, default=300_000)
    ap.add_argument("--objective", default="hard-neg", choices=["hard-neg", "cosine-lin"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    spec = ChunkAttnEncoderSpec(
        source_embedding_config_id=args.source_config,
        out_path=Path(args.out),
        dict_dim=args.dict_dim,
        top_k=args.top_k,
        att_dim=args.att_dim,
        heads=args.heads,
        cap_chunks=args.cap_chunks,
        reference_n=args.reference_n,
        epochs=args.epochs,
        train_pairs=args.train_pairs,
        objective=args.objective,
        seed=args.seed,
    )
    train_chunk_attn_encoder(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
