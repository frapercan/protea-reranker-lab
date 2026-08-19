"""Training over a corpus that does not fit in the autograd graph.

Chunking the forward is enough to ENCODE any corpus: with no gradient the peak is
one chunk plus the accumulator, measured flat near 2 GB from 400,000 residues to
6,400,000. Training is not: autograd retains each chunk's OUTPUT because the
accumulator needs it, so the peak grows with the total. Gradient checkpointing
does not help either, measured at 5.95 GB against 5.75 without, because what is
retained is the outputs rather than the intermediates.

Splitting the backward in two fixes it, and the load-bearing test is that it
produces the SAME gradient. A wrong chain rule here does not raise; it trains to
something plausible. The derivative that has to be right is that the pooled code
is the MEAN of its residues, so each chunk's backward seed is the protein's
gradient divided by its residue count.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from protea_reranker_lab.chunked_backward import (
    ChunkedFit,
    accumulate_gradient,
    fit_step,
    pooled_codes,
)


def _setup(n_proteins=12, per=40, d=16, width=32, chunk=100, keep=None, seed=0):
    torch.manual_seed(seed)
    n = n_proteins * per
    x = torch.randn(n, d)
    owner = torch.repeat_interleave(torch.arange(n_proteins), torch.full((n_proteins,), per))
    counts = torch.full((n_proteins,), float(per))
    w = (torch.randn(d, width) * d ** -0.5).requires_grad_()

    def project(block, weights, k):
        out = block @ weights
        if k is None:
            return out
        kth = out.abs().kthvalue(out.shape[-1] - k + 1, dim=-1, keepdim=True).values
        return out.masked_fill(out.abs() < kth, 0.0)

    def chunks():
        for s in range(0, n, chunk):
            yield x[s:s + chunk], owner[s:s + chunk]

    fit = ChunkedFit(chunks, w, counts, n_proteins, width, project, keep)
    return fit, x, owner, counts, w, project


def _naive(x, owner, counts, w, project, n_proteins, width, keep, loss_fn):
    out = project(x, w, keep)
    sums = torch.zeros(n_proteins, width).index_add(0, owner, out)
    loss = loss_fn(sums / counts.unsqueeze(1))
    loss.backward()
    return float(loss), w.grad.clone()


# --------------------------------------------------------------- the pooled aggregate

def test_the_chunked_aggregate_equals_the_whole_one():
    fit, x, owner, counts, w, project = _setup()

    chunked = pooled_codes(fit, torch)
    whole = torch.zeros(12, 32).index_add(0, owner, project(x, w, None)) / counts.unsqueeze(1)

    assert torch.allclose(chunked, whole, atol=1e-5)


def test_the_aggregate_retains_no_graph():
    """Phase one exists to hold nothing, so a tensor that still carried a graph
    would defeat the whole design without failing."""
    fit, *_ = _setup()

    assert pooled_codes(fit, torch).grad_fn is None


# ------------------------------------------------------------------- the gradient

@pytest.mark.parametrize("keep", [None, 4])
def test_the_two_phase_gradient_matches_the_naive_one(keep):
    """The load-bearing test. A wrong chain rule trains to something plausible."""
    fit, x, owner, counts, w, project = _setup(keep=keep)
    target = torch.randn(12, 32)
    loss_fn = lambda p: ((p - target) ** 2).mean()

    chunked_loss = fit_step(fit, loss_fn, torch)
    g_chunked = w.grad.clone()
    w.grad = None
    naive_loss, g_naive = _naive(x, owner, counts, w, project, 12, 32, keep, loss_fn)

    assert chunked_loss == pytest.approx(naive_loss, rel=1e-6)
    assert (g_naive - g_chunked).norm() / g_naive.norm() < 1e-5


def test_the_seed_is_divided_by_the_residue_count():
    """The derivative of a MEAN. Forgetting the division scales the gradient by the
    protein's length, which trains, converges, and is wrong by a factor nobody sees."""
    fit, x, owner, counts, w, project = _setup(n_proteins=4, per=25, chunk=50)
    seed = torch.ones(4, 32)

    accumulate_gradient(fit, seed, torch)
    scaled = w.grad.clone()
    w.grad = None
    accumulate_gradient(ChunkedFit(fit.chunks, w, counts * 2, 4, 32, project, None),
                        seed, torch)

    assert torch.allclose(scaled, w.grad * 2, atol=1e-6)


# ---------------------------------------------------------------------- the factory

def test_an_exhausted_iterator_would_be_caught_by_the_factory_contract():
    """chunks is a factory because the second walk needs the same sequence. Handing
    an iterator instead yields nothing on the second pass and leaves a zero
    gradient, which reads as convergence."""
    fit, *_ = _setup()
    first = [rows.shape[0] for _, rows in fit.chunks()]
    second = [rows.shape[0] for _, rows in fit.chunks()]

    assert first == second and sum(first) > 0


def test_both_walks_see_every_residue():
    fit, x, *_ = _setup()

    seen = sum(block.shape[0] for block, _ in fit.chunks())

    assert seen == x.shape[0]
