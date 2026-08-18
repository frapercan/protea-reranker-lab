"""Fitting one or two projections, jointly or in stages, and encoding with them.

Fit dense and sparsify afterwards, which is a decision rather than a convenience:
one fit per dictionary width then serves every k, and nothing in the loop has to
survive a non-differentiable top-k.

JOINT AGAINST STAGED, WHICH IS THE QUESTION THIS EXISTS TO ANSWER

Joint is one forward and one loss: residues go through the residue map, the codes
are aggregated, the protein map runs on the aggregate, and the gradient reaches
both. The residue map is then shaped entirely by a protein-level signal averaged
over hundreds of residues, which is a weak and indirect gradient per residue.

Staged fits the residue map alone first, against the same protein-level target but
with no protein map above it, freezes it, and fits the protein map on top. The
residue map then has to be useful on its own before anything is built over it,
which is a stronger constraint and may be the wrong one.

Neither is obviously better and the difference is measurable, so it is a field.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from protea_reranker_lab.architecture import Architecture

log = logging.getLogger(__name__)


@dataclass
class FittedMaps:
    """The learned matrices, kept as numpy so encoding needs no framework."""

    residue: np.ndarray | None
    protein: np.ndarray | None
    epochs: int
    final_loss: float


def _targets(closures: list[frozenset[str]], pairs: np.ndarray) -> np.ndarray:
    """Jaccard of the two closures, which is what the code cosine must match."""
    out = np.empty(pairs.shape[0], dtype=np.float32)
    for n, (i, j) in enumerate(pairs):
        a, b = closures[i], closures[j]
        union = len(a | b)
        out[n] = len(a & b) / union if union else 0.0
    return out


def _residue_topk(t, keep: int, torch):
    """Keep the ``keep`` largest magnitudes per residue, zero the rest.

    This is the nonlinearity that makes the granularity axis exist at all. Without
    it the residue map is linear, projecting and averaging commute exactly, and
    "project each residue then aggregate" and "aggregate then project" are the same
    function: measured on the toy they agreed to 2e-6 on values of scale 1.7, so
    every architecture arm returned the same number and the comparison was empty.

    Differentiable in the way that matters: the selection is piecewise constant and
    the gradient reaches the selected entries, so the map is trained knowing it will
    be truncated rather than truncated after being trained dense. That is a real
    departure from fit-dense-then-sparsify, and it is confined to the residue stage
    because that is the only stage where the truncation has to happen before an
    aggregation.
    """
    if keep >= t.shape[-1]:
        return t
    # A threshold, not a mask, and the magnitudes computed ONCE. The obvious form
    # builds a zeros_like, scatters ones into it and multiplies, holding three
    # tensors of the residue tensor's size, which is what ran the card out of
    # memory on the longest band. Taking abs twice, as the first fix did, still
    # peaked at 2.3 times the input; reusing it peaks near 1.3.
    magnitude = t.abs()
    kth = magnitude.kthvalue(t.shape[-1] - keep + 1, dim=-1, keepdim=True).values
    magnitude = magnitude < kth
    return t.masked_fill(magnitude, 0.0)


#: Residues processed in one matmul. The batch is bounded by RESIDUES rather than
#: by proteins because lengths span 40 to 8,886 here: a fixed protein count makes
#: the working set swing by two orders of magnitude and the card runs out on the
#: unlucky batch.
RESIDUE_BATCH = 200_000


class ResidueCache:
    """Residues concatenated and moved to the device ONCE, with the owner map.

    The training loop projects the same residues every epoch and only the matrix
    changes, so concatenating and transferring per epoch pays the same cost 250
    times. Measured on a 200,000-residue batch: concatenate 0.095 s, transfer
    0.087 s, matmul 0.081 s, so two thirds of the work was moving data that had
    not changed.

    A CAUTION THAT IS NOT ABOUT SPEED. Batching changes the matmul's accumulation
    order, which moves activations by about 1e-7, and top-k turns that into a
    DIFFERENT ATOM whenever two are near-tied. One protein in four hundred selected
    a different atom and its pooled vector then differed by the full magnitude of
    the swap. The sparse code is not numerically stable under changes in the
    compute path, which is a property of the approach rather than of this cache,
    and it means a code produced on one backend cannot be assumed bit-identical to
    one produced on another.
    """

    def __init__(self, residues: list[np.ndarray], torch, device):
        lengths = [r.shape[0] for r in residues]
        self.matrix = torch.tensor(np.concatenate(residues, axis=0), device=device)
        self.owner = torch.repeat_interleave(
            torch.arange(len(residues), device=device),
            torch.tensor(lengths, device=device),
        )
        self.counts = torch.tensor(lengths, device=device, dtype=torch.float32).unsqueeze(1)
        self.n = len(residues)

    def pooled(self, weights, torch, keep: int | None = None):
        block = self.matrix if weights is None else self.matrix @ weights
        if weights is not None and keep is not None:
            block = _residue_topk(block, keep, torch)
        sums = torch.zeros(self.n, block.shape[1], device=block.device, dtype=block.dtype)
        sums.index_add_(0, self.owner, block)
        return sums / self.counts

    def bytes(self) -> int:
        return self.matrix.numel() * 4


def _pooled_batched(residues: list[np.ndarray], matrix, torch, device,
                    keep: int | None = None):
    """Mean per protein, computed with ONE matmul per batch instead of one per protein.

    The loop this replaces did a small matmul per protein and left the card at one
    per cent while a Python loop saturated a core. At 400 proteins that is minutes;
    at 20,000 it is the whole cost, since the loop runs every epoch.

    Residues are concatenated into a single matrix, projected once, sparsified once
    if asked, and then summed back per protein through ``index_add_``, which is the
    segment reduction the per-protein mean actually is. No padding, so a batch of
    forty-residue proteins costs forty rows rather than a padded eight thousand.
    """
    lengths = [r.shape[0] for r in residues]
    out = torch.zeros(len(residues), device=device,
                      dtype=torch.float32) if False else None
    pooled = []
    start = 0
    while start < len(residues):
        end, total = start, 0
        while end < len(residues) and (total + lengths[end] <= RESIDUE_BATCH or end == start):
            total += lengths[end]
            end += 1
        block = torch.tensor(np.concatenate(residues[start:end], axis=0), device=device)
        if matrix is not None:
            block = block @ matrix
            if keep is not None:
                block = _residue_topk(block, keep, torch)
        owner = torch.repeat_interleave(
            torch.arange(end - start, device=device),
            torch.tensor(lengths[start:end], device=device),
        )
        sums = torch.zeros(end - start, block.shape[1], device=device, dtype=block.dtype)
        sums.index_add_(0, owner, block)
        counts = torch.tensor(lengths[start:end], device=device,
                              dtype=block.dtype).unsqueeze(1)
        pooled.append(sums / counts)
        start = end
    return torch.cat(pooled, dim=0)


def _pooled(residues: list[np.ndarray], matrix, torch, device, keep: int | None = None):
    """Mean over residues, projected and optionally sparsified first.

    With ``keep`` set the mean runs over TOP-K CODES rather than dense ones, so the
    aggregate is a magnitude-weighted usage histogram over the dictionary: entry j
    is how often atom j was selected locally times how strongly. That is a
    different object from the mean of dense projections, where an atom moderately
    present everywhere and an atom intense in one place are blended before anything
    is selected.
    """
    out = []
    for r in residues:
        t = torch.tensor(r, device=device)
        if matrix is not None:
            t = t @ matrix
            if keep is not None:
                t = _residue_topk(t, keep, torch)
        out.append(t.mean(dim=0))
    return torch.stack(out)


@dataclass(frozen=True)
class FitConfig:
    """How the fit runs, as an object so a result can record the whole recipe.

    ``balance`` weights the load-balancing term that asks the dictionary to use
    itself. It acts on mean absolute activation, which is what a later top-k reads,
    and is zero by default so the baseline arm is the champion's objective exactly.
    """

    epochs: int = 200
    n_pairs: int = 40000
    seed: int = 42
    lr: float = 1e-3
    balance: float = 0.0


def _sample_pairs(n: int, cfg: FitConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    i = rng.integers(0, n, size=cfg.n_pairs)
    j = rng.integers(0, n, size=cfg.n_pairs)
    keep = i != j
    return np.stack([i[keep], j[keep]], axis=1)


def fit_architecture(residues: dict[str, np.ndarray], closures: dict[str, frozenset[str]],
                     arch: Architecture, cfg: FitConfig | None = None) -> FittedMaps:
    """Fit whichever projections the architecture declares."""
    import torch

    cfg = cfg or FitConfig()
    epochs, seed, lr, balance = cfg.epochs, cfg.seed, cfg.lr, cfg.balance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accessions = sorted(residues)
    stack = [residues[a] for a in accessions]
    closure_list = [closures[a] for a in accessions]
    dim = stack[0].shape[1]

    pairs = _sample_pairs(len(accessions), cfg)
    y = torch.tensor(_targets(closure_list, pairs), device=device)
    ti = torch.tensor(pairs[:, 0], device=device)
    tj = torch.tensor(pairs[:, 1], device=device)

    res_w, prot_w = _init_maps(arch, dim, seed, torch, device)

    keep = arch.residue.keep if (arch.residue and not arch.residue.is_dense) else None
    cache = ResidueCache(stack, torch, device)
    log.info("residue cache: %.2f GB on %s", cache.bytes() / 1024 ** 3, device)

    def run(params, use_protein: bool):
        pooled = cache.pooled(res_w if arch.residue else None, torch, keep)
        z = pooled @ prot_w if (use_protein and prot_w is not None) else pooled
        zi, zj = z[ti], z[tj]
        cos = (zi * zj).sum(1) / (zi.norm(dim=1) * zj.norm(dim=1) + 1e-8)
        loss = ((cos - y) ** 2).mean()
        if balance > 0:
            mass = z.abs().mean(dim=0)
            p = mass / (mass.sum() + 1e-9)
            loss = loss + balance * p.numel() * (p * p).sum()
        return loss

    def optimise(params, use_protein: bool, n: int) -> float:
        opt = torch.optim.Adam(params, lr=lr)
        loss = torch.zeros(())
        for e in range(n):
            opt.zero_grad()
            loss = run(params, use_protein)
            loss.backward()
            opt.step()
            if e % 50 == 0:
                log.debug("  %s epoch %d loss %.5f", arch.name, e, float(loss))
        return float(loss)

    final = _run_schedule(arch, res_w, prot_w, optimise, epochs)
    if arch.stages == 2 and not arch.joint_training:
        res_w = res_w.detach()

    return FittedMaps(
        residue=res_w.detach().cpu().numpy() if res_w is not None else None,
        protein=prot_w.detach().cpu().numpy() if prot_w is not None else None,
        epochs=epochs, final_loss=final,
    )


def _init_maps(arch: Architecture, dim: int, seed: int, torch, device):
    """Both matrices, scaled so the code's variance does not depend on the width.

    Scaled by the fan-in rather than left at unit variance, because the dictionary
    width is an axis in this study: an unscaled init would make a wide dictionary
    start with larger activations than a narrow one and confound width with
    initialisation.
    """
    torch.manual_seed(seed)
    res_w = (torch.randn(dim, arch.residue.width, device=device) * dim ** -0.5
             ).requires_grad_() if arch.residue else None
    in_dim = arch.residue.width if arch.residue else dim
    prot_w = (torch.randn(in_dim, arch.protein.width, device=device) * in_dim ** -0.5
              ).requires_grad_() if arch.protein else None
    return res_w, prot_w


def _run_schedule(arch: Architecture, res_w, prot_w, optimise, epochs: int) -> float:
    """One fit, or two in sequence when the architecture asks for stages.

    Staged is not a smaller version of joint: the residue map is optimised with no
    protein map above it, so it has to be useful on its own before anything is
    built over it. That is a stronger constraint and may be the wrong one, which is
    why it is measured rather than assumed.
    """
    if arch.stages == 2 and not arch.joint_training:
        optimise([res_w], use_protein=False, n=epochs)
        return optimise([prot_w], use_protein=True, n=epochs)
    params = [p for p in (res_w, prot_w) if p is not None]
    return optimise(params, use_protein=True, n=epochs)


def encode(residues: dict[str, np.ndarray], arch: Architecture,
           maps: FittedMaps) -> tuple[np.ndarray, list[str]]:
    """Protein codes, dense, in accession order. Sparsification happens downstream."""
    accessions = sorted(residues)
    keep = arch.residue.keep if (arch.residue and not arch.residue.is_dense) else None
    out = []
    for a in accessions:
        x = residues[a]
        if maps.residue is not None:
            x = x @ maps.residue
            if keep is not None:
                m = np.zeros_like(x)
                idx = np.argpartition(-np.abs(x), keep - 1, axis=1)[:, :keep]
                np.put_along_axis(m, idx, 1.0, axis=1)
                x = x * m
        v = x.mean(axis=0)
        if maps.protein is not None:
            v = v @ maps.protein
        out.append(v)
    return np.vstack(out).astype(np.float32), accessions
