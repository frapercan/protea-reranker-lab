"""GCN GO-label encoder (Phase 1 of the BP structural lever).

Replaces the FIXED GO codes (the frozen `GO_V` label tower of the two-tower) with a
LEARNABLE DAG-aware label encoder. Term embeddings are initialised from the fused
sparse functional GO codes (whitened-BioBERT text k-WTA + t0 co-annotation PPMI/SVD
k-WTA, 1024-d) and message-passed over the GO `is_a` DAG (`go_parents`, bidirectional
child<->parent + self loops, symmetric-normalised adjacency, a few residual layers) so
rare/deep BP terms inherit neighbour signal.

The joint scorer keeps the two-tower geometry:
    score(prot, term) = temp * <proj(prot_code), gcn_emb[term]> + bias[term]
where `proj` is the protein tower (ProjHead from p2/two_tower.py) and `gcn_emb` is the
GCN label tower. Both towers + temperature + bias train end-to-end (ASL + DAG hinge).

Temporal honesty: the GCN INPUT co-annotation block is per-cut (build_per_cut_codes.py,
frozen v227 SVD basis). The trained head is a fixed method (like the frozen PLM / the
d8979601 encoder); only the per-cut INPUT codes vary at scoring time -- no future-snapshot
codes touch a training row, exactly the Phase 0 contract.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(edges, V, device):
    """Symmetric-normalised adjacency D^-1/2 (A + I) D^-1/2 over the bidirectional DAG.

    edges: (2, E) int array, edges[0]=child vocab idx, edges[1]=parent vocab idx.
    """
    c = np.asarray(edges[0]); p = np.asarray(edges[1])
    self_i = np.arange(V)
    src = np.concatenate([c, p, self_i])      # message source
    dst = np.concatenate([p, c, self_i])      # message destination
    idx = torch.tensor(np.stack([dst, src]), dtype=torch.long)
    val = torch.ones(idx.shape[1], dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, (V, V)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    r, cc = A.indices()
    v = A.values() * dinv[r] * dinv[cc]
    return torch.sparse_coo_tensor(A.indices(), v, (V, V)).coalesce().to(device)


class LabelGCN(nn.Module):
    """DAG-aware GO-label encoder. init (V x d_in) -> emb (V x d_out), L2-normalised
    to the sqrt(2) scale of the original fused GO codes."""

    def __init__(self, d_in=1024, d_hid=1024, d_out=1024, layers=2, dropout=0.0):
        super().__init__()
        dims = [d_in] + [d_hid] * (layers - 1) + [d_out]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layers = layers

    def forward(self, init, A):
        h = init
        for i, lin in enumerate(self.lins):
            agg = torch.sparse.mm(A, h)
            z = lin(agg)
            if i < self.layers - 1:
                z = F.gelu(z)
                z = self.drop(z)
                if z.shape == h.shape:
                    z = z + h          # residual on the hidden layers
            h = z
        return F.normalize(h, dim=1) * (2.0 ** 0.5)


def load_init_aligned(npz_path, vocab_go):
    """Load a go_sparse_codes_v{N}.npz and align its rows to the two-tower vocab order."""
    d = np.load(npz_path, allow_pickle=True)
    ids = [str(x) for x in d["go_ids"]]
    idmap = {g: i for i, g in enumerate(ids)}
    rows = np.array([idmap[g] for g in vocab_go], dtype=np.int64)
    return d["codes"][rows].astype(np.float32)
