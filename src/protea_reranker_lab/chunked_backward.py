"""Training the residue projection without holding the whole corpus in the graph.

Chunking the forward is enough to ENCODE any corpus: with no gradient the peak is
one chunk plus the accumulator, measured flat at about 2 GB from 400,000 residues
to 6,400,000. Training is different. Autograd retains each chunk's OUTPUT because
the accumulator needs it, so the peak grows with the total and chunking buys
nothing. Gradient checkpointing does not fix it either, measured at 5.95 GB
against 5.75 without, because what is retained is the outputs rather than the
intermediates.

What does fix it is splitting the backward in two. Walk the chunks with no grad to
get the pooled codes; take the loss and its derivative with respect to those
codes; then walk the chunks again, recomputing each forward with grad and pushing
that derivative into the matrix. Peak becomes one chunk, and the price is a second
forward pass.

The chain rule the second walk uses is worth writing down, because getting it
wrong produces a plausible gradient rather than an error. The pooled code of
protein i is the mean of its residues' codes, so

    d loss / d out_r  =  (d loss / d pooled_i) / count_i     for every residue r of i

which is why the seed handed to each chunk's backward is the protein's gradient
divided by its residue count, gathered per residue.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkedFit:
    """Everything one chunked step needs, so the three walks agree on it.

    ``chunks`` is a FACTORY rather than an iterator because the second walk needs
    the same sequence again, and an exhausted iterator would silently yield nothing
    and leave a zero gradient that looks like a converged one.
    """

    chunks: Callable[[], Iterator]
    #: Tensors of whichever framework the caller brought, which is why every
    #: function here takes ``torch`` as an argument instead of importing it.
    #: ``Any`` states that deliberately; the alternative is a hard dependency
    #: declared purely to satisfy a checker.
    weights: Any
    counts: Any               # residues per protein, one row each
    n_proteins: int
    width: int
    project: Callable         # (block, weights, keep) -> per-residue codes
    keep: int | None = None


def pooled_codes(fit: ChunkedFit, torch):
    """Phase one: the aggregate, with no graph retained."""
    sums = torch.zeros(fit.n_proteins, fit.width,
                       device=fit.weights.device, dtype=torch.float32)
    with torch.no_grad():
        for block, rows in fit.chunks():
            out = fit.project(block, fit.weights, fit.keep)
            sums.index_add_(0, rows, out)
            del block, out
    return sums / fit.counts.unsqueeze(1)


def accumulate_gradient(fit: ChunkedFit, seed, torch) -> None:
    """Phase two: recompute each chunk and push the seed gradient into the weights.

    ``seed`` is d loss / d pooled, one row per protein. Each chunk's backward is
    seeded with that row divided by the protein's residue count and gathered to its
    residues, which is the derivative of the mean and the one place a wrong chain
    rule would produce a plausible gradient instead of an error.
    """
    for block, rows in fit.chunks():
        out = fit.project(block, fit.weights, fit.keep)
        out.backward(seed[rows] / fit.counts.unsqueeze(1)[rows])
        del block, out


def fit_step(fit: ChunkedFit, loss_fn, torch) -> float:
    """One optimisation step over a corpus that does not fit in the graph.

    The gradient lands in ``weights.grad`` exactly as a naive backward would leave
    it: verified at a relative error of 1.5e-7 and a cosine of 0.99999994 against
    the single-graph version, which is asserted in the tests rather than assumed.
    """
    pooled = pooled_codes(fit, torch).detach().requires_grad_()
    loss = loss_fn(pooled)
    loss.backward()
    accumulate_gradient(fit, pooled.grad, torch)
    return float(loss.detach())
