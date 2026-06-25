"""Learned attention-pool aggregation for the SDR substrate x aggregation factorial.

The naive aggregation (``sdr.py`` + ``run_sdr_length_stratified.py``) sparsifies each unit
(chunk / residue) independently and then bundles the supports by vote / OR. That throws away
WHICH units matter for function. This module is the LEARNED counterpart: a small attention pool
weights the units, the pooled vector is projected into a sparse GO-aligned code, and the whole
thing is trained with the same hard-negative Lin-contrastive objective the champion mean encoder
uses (``encoder_ablation.fit_encoder``). The decisive, never-tested cell is chunk/residue + this
learned pool, where locality should matter.

Architecture (deliberately small / fast, RTX-3060-safe)::

    units (n_u, d) --attn--> pooled (d) --Linear(d->dict)--> z (dict) --top-k real--> code

Attention is a single-head additive pool with a learnable query::

    s_u = v . tanh(W u + b);  a = softmax(s);  pooled = sum_u a_u * u

Training pools per protein (variable n_u) inside the contrastive loop, so the attention learns
which units carry the function signal under the SAME supervision as the dense champion.

torch is imported lazily inside the functions; annotations are guarded by ``TYPE_CHECKING`` so the
module stays importable for autodoc / unit-test collection without the GPU stack (lab landmine).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol, Sequence

import numpy as np

from protea_reranker_lab.sdr import GoDag, information_content, lin_pairwise

if TYPE_CHECKING:  # annotations only; torch is the heavy GPU dep, imported lazily below
    import torch

    class AttnEncoder(Protocol):
        """Static view of the attention-pool encoder (a real ``nn.Module`` at runtime)."""

        def encode(self, unit_list: "list[torch.Tensor]") -> "torch.Tensor": ...

        def to(self, device: str) -> "AttnEncoder": ...

        def eval(self) -> "AttnEncoder": ...

        def parameters(self) -> "Iterator[torch.Tensor]": ...

log = logging.getLogger("sdr-pool")


@dataclass
class PoolSpec:
    """Hyper-parameters for one learned attention-pool encoder."""
    dict_dim: int = 2048
    top_k: int = 128
    attn_dim: int = 256
    epochs: int = 120
    train_pairs: int = 200_000
    hardneg_anchors: int = 1500
    hardneg_knn: int = 30
    lr: float = 1e-3
    seed: int = 42
    # memory bounds (RTX-3060 12GB): per epoch pool a SUBSET of proteins with grad and use only
    # the pairs fully inside that subset (minibatch contrastive). max_units caps the residues per
    # protein fed to the pool (longest proteins are subsampled by stride), so the autograd graph
    # over per-residue activations stays bounded regardless of sequence length.
    proteins_per_step: int = 1200
    max_units: int = 512


# --------------------------------------------------------------------------- numeric utils
def l2n(X: np.ndarray) -> np.ndarray:
    """L2-normalise rows of a (n, d) matrix (zero rows left as zero)."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def topk_real(X: np.ndarray, k: int) -> np.ndarray:
    """Keep the top-k by |value| per row, zero the rest (the sparse real code)."""
    if k >= X.shape[1]:
        return X.copy()
    out = np.zeros_like(X)
    idx = np.argpartition(-np.abs(X), k, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(X, idx, axis=1), axis=1)
    return out


def cap_units(units: np.ndarray, max_units: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample a protein's (n_u, d) unit matrix to at most ``max_units`` rows.

    Long proteins (many residues) are bounded by an even stride so the attention pool still sees
    the whole sequence at a fixed memory budget; short proteins (<= max_units) pass through.
    """
    n_u = units.shape[0]
    if n_u <= max_units:
        return units
    idx = np.linspace(0, n_u - 1, max_units).round().astype(np.int64)
    return units[idx]


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Sample ``n_pairs`` distinct unordered index pairs from ``range(n)``."""
    n_pairs = min(n_pairs, n * (n - 1) // 2)
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


# --------------------------------------------------------------------------- model
def _build_module(in_dim: int, spec: PoolSpec) -> "AttnEncoder":
    """Construct the attention-pool + projection module (additive single-head attention)."""
    import torch.nn as nn

    class AttnPoolEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W = nn.Linear(in_dim, spec.attn_dim)
            self.v = nn.Linear(spec.attn_dim, 1, bias=False)
            self.proj = nn.Linear(in_dim, spec.dict_dim)

        def pool(self, units: "torch.Tensor") -> "torch.Tensor":
            """units: (n_u, d) for ONE protein -> pooled (d,)."""
            import torch

            scores = self.v(torch.tanh(self.W(units))).squeeze(-1)  # (n_u,)
            attn = torch.softmax(scores, dim=0)                      # (n_u,)
            return (attn.unsqueeze(-1) * units).sum(0)              # (d,)

        def encode(self, unit_list: "list[torch.Tensor]") -> "torch.Tensor":
            """unit_list: list of (n_u, d) -> z (n, dict)."""
            import torch

            pooled = torch.stack([self.pool(u) for u in unit_list], dim=0)  # (n, d)
            return self.proj(pooled)

    return AttnPoolEncoder()


# --------------------------------------------------------------------------- training
def fit_attention_pool(
    unit_arrays: Sequence[np.ndarray],
    closures: Sequence[frozenset[str]],
    dag: GoDag,
    spec: PoolSpec,
    mean_for_mining: np.ndarray,
) -> "AttnEncoder":
    """Train the attention-pool encoder with the hard-neg Lin-contrastive objective.

    ``unit_arrays[i]`` is the (n_u, d) L2-normalised unit matrix (chunks or residues) of protein
    ``i``; ``closures[i]`` its propagated GO closure; ``mean_for_mining`` the (n, d) per-protein
    mean used ONLY to mine embedding-near / GO-far hard-negative pairs (same recipe as the dense
    champion). Returns the trained module on its device.
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(spec.seed)
    n = len(closures)
    d = unit_arrays[0].shape[1]

    ic = information_content(closures, dag)
    bic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in closures]

    pairs = sample_pairs(n, spec.train_pairs, rng)
    # hard-neg mining: embedding-near pairs whose true Lin is often low (the hard cases)
    Rn = l2n(mean_for_mining)
    n_anchor = min(spec.hardneg_anchors, n)
    anchors = rng.choice(n, size=n_anchor, replace=False)
    S = Rn[anchors] @ Rn.T
    np.put_along_axis(S, anchors[:, None], -1.0, axis=1)
    knn = min(spec.hardneg_knn, n - 1)
    nbr = np.argpartition(-S, knn, axis=1)[:, :knn]
    for a_i, anchor in enumerate(anchors):
        for j in nbr[a_i]:
            pairs.append((int(anchor), int(j)) if anchor < j else (int(j), int(anchor)))

    y_all = np.asarray(lin_pairwise(closures, ic, pairs, bic), dtype=np.float32)
    # index pairs by protein for the minibatch step (only pairs fully inside the step are scored)
    pairs_arr = np.asarray(pairs, dtype=np.int64)  # (P, 2)
    y_arr = y_all

    # units stay on CPU (huge for residues); each step moves only its protein-batch to GPU, with
    # a per-protein unit cap so the autograd graph over activations is bounded by length too.
    units_cpu = [cap_units(u.astype(np.float32), spec.max_units) for u in unit_arrays]
    enc = _build_module(d, spec).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=spec.lr)

    step_n = min(spec.proteins_per_step, n)
    for e in range(spec.epochs):
        sel = rng.choice(n, size=step_n, replace=False)
        sel_set = {int(s): i for i, s in enumerate(sel)}  # global idx -> local idx in the step
        # pairs fully inside the selected protein subset
        mask = np.fromiter((p0 in sel_set and p1 in sel_set for p0, p1 in pairs_arr),
                           dtype=bool, count=len(pairs_arr))
        if not mask.any():
            continue
        sub_pairs = pairs_arr[mask]
        li = torch.tensor([sel_set[int(p)] for p in sub_pairs[:, 0]], device=dev)
        lj = torch.tensor([sel_set[int(p)] for p in sub_pairs[:, 1]], device=dev)
        y = torch.tensor(y_arr[mask], device=dev)

        batch_units = [torch.tensor(units_cpu[int(s)], device=dev) for s in sel]
        z = enc.encode(batch_units)  # (step_n, dict) with grad through the pool
        zi, zj = z[li], z[lj]
        cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
        loss = ((cos - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 30 == 0:
            log.info("  [attn-pool] epoch %3d loss=%.4f (step_n=%d pairs=%d)",
                     e, float(loss.detach()), step_n, int(mask.sum()))
        del z, batch_units, loss
    return enc


def apply_attention_pool(
    enc: "AttnEncoder", unit_arrays: Sequence[np.ndarray], top_k: int,
    max_units: int = 512, batch: int = 256,
) -> np.ndarray:
    """Encode proteins through the trained pool; return the (n, dict) top-k real codes.

    Units are capped to ``max_units`` per protein (same bound as training) and moved to GPU in
    small batches under ``no_grad`` so inference stays within the 12GB budget for residues.
    """
    import torch

    dev = next(enc.parameters()).device
    out: list[np.ndarray] = []
    enc.eval()
    with torch.no_grad():
        for b in range(0, len(unit_arrays), batch):
            chunk = [torch.tensor(cap_units(u.astype(np.float32), max_units), device=dev)
                     for u in unit_arrays[b:b + batch]]
            z = enc.encode(chunk).cpu().numpy().astype(np.float32)
            out.append(topk_real(z, top_k))
    return np.vstack(out)
