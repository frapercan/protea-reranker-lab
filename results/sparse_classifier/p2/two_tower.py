"""Two-tower sparse functional GO candidate generator (PROTEA iteration `sparse-classifier`).

PROTEIN tower input  : frozen learned-champion d8979601 k-WTA codes (2048-d, ~6% active).
GO tower (frozen)    : sparse functional codes (1024-d, whitened-BioBERT-text k-WTA + t0
                       co-annotation PPMI/SVD k-WTA), L2 norm sqrt(2).
Learned head         : projects protein code (2048) -> GO-code space (1024).
score(prot, term)    = <proj(prot_code), go_code[term]> over the frozen GO vocab codes.
top-100              = top-k terms by score => candidate generator output.

Loss = ASL (asymmetric multi-label) over the vocab + soft DAG-consistency hinge
(score(child) <= score(parent) on go_parents is_a edges).

No live DB; everything is loaded from the cached prep.npz (built from v227 t0 files).
"""
import os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- model -----------------------------
class ProjHead(nn.Module):
    """Protein-code -> GO-code-space projection + learnable per-term bias (frequency prior) and
    a temperature. Linear or 1-hidden MLP, optional query k-WTA.

    score(prot, term) = temp * <proj(prot_code), go_code[term]> + bias[term]
    The bias is a per-term constant (same for every protein) = a learned frequency prior; it does
    not break the two-tower retrieval geometry, it only recalibrates the per-term threshold.
    """
    def __init__(self, d_in=2048, d_out=1024, V=0, hidden=0, kwta=0, dropout=0.0):
        super().__init__()
        self.kwta = kwta
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, d_out),
            )
        else:
            self.net = nn.Linear(d_in, d_out)
        self.bias = nn.Parameter(torch.zeros(V)) if V else None
        self.log_temp = nn.Parameter(torch.zeros(()))  # temperature, init 1.0

    def project(self, x):
        z = self.net(x)
        if self.kwta and self.kwta > 0:
            k = self.kwta
            thresh = z.abs().topk(k, dim=1).values[:, -1:].detach()
            z = z * (z.abs() >= thresh)
        return z

    def forward(self, x, GO_T):
        z = self.project(x)
        logits = self.log_temp.exp() * (z @ GO_T)
        if self.bias is not None:
            logits = logits + self.bias
        return logits


# ----------------------------- losses -----------------------------
def asl_loss(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
    """Asymmetric multi-label loss over the full vocab (dense targets)."""
    x_sig = torch.sigmoid(logits)
    xs_pos = x_sig
    xs_neg = 1.0 - x_sig
    if clip and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    los_pos = targets * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg
    pt0 = xs_pos * targets
    pt1 = xs_neg * (1 - targets)
    pt = pt0 + pt1
    gamma = gamma_pos * targets + gamma_neg * (1 - targets)
    w = torch.pow(1 - pt, gamma)
    loss = loss * w
    return -loss.sum() / logits.shape[0]


def dag_hinge(logits, edges_child, edges_parent, margin=0.0):
    """score(child) <= score(parent): penalise relu(s_child - s_parent + margin) over sampled edges."""
    s_child = logits[:, edges_child]
    s_parent = logits[:, edges_parent]
    return F.relu(s_child - s_parent + margin).mean()


# ----------------------------- data -----------------------------
class Data:
    def __init__(self, prep_path, prot_codes_path):
        d = np.load(prep_path, allow_pickle=True)
        self.GO_V = torch.from_numpy(d["GO_V"]).to(DEV)          # V x 1024 (frozen)
        self.indptr = d["indptr"]; self.indices = d["indices"]    # CSR positives (vocab space)
        self.tf = d["tf"]                                          # term freq baseline
        self.edges = torch.from_numpy(d["edges"].astype(np.int64))
        self.train_idx = d["train_idx"]; self.test_idx = d["test_idx"]
        self.vocab_go = d["vocab_go"]
        self.V = self.GO_V.shape[0]
        p = np.load(prot_codes_path, allow_pickle=True)
        self.accs = p["accs"]
        self.codes = torch.from_numpy(p["codes"].astype(np.float32)).to(DEV)  # N x 2048 (frozen)

    def positives(self, i):
        return self.indices[self.indptr[i]:self.indptr[i + 1]]


def build_target(data, batch_idx):
    """Dense V-wide multi-label target for a batch of protein row indices."""
    B = len(batch_idx)
    tgt = torch.zeros(B, data.V, device=DEV)
    for b, i in enumerate(batch_idx):
        pos = data.positives(i)
        if len(pos):
            tgt[b, pos] = 1.0
    return tgt


# ----------------------------- train one seed -----------------------------
def train_seed(data, seed, train_idx, cfg, log_prefix=""):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ProjHead(2048, 1024, V=data.V, hidden=cfg["hidden"], kwta=cfg["kwta"],
                     dropout=cfg["dropout"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    GO_T = data.GO_V.t().contiguous()                         # 1024 x V
    ec = data.edges[0].to(DEV); ep = data.edges[1].to(DEV)
    n_edge = ec.shape[0]
    losses = []
    for ep_i in range(cfg["epochs"]):
        model.train()
        rng = np.random.default_rng(seed * 1000 + ep_i)
        order = rng.permutation(len(train_idx))
        tot = 0.0; nb = 0
        for s in range(0, len(order), cfg["bs"]):
            bidx = train_idx[order[s:s + cfg["bs"]]]
            x = data.codes[bidx]
            tgt = build_target(data, bidx)
            logits = model(x, GO_T)                            # B x V
            loss = asl_loss(logits, tgt, cfg["gn"], cfg["gp"], cfg["clip"])
            if cfg["dag_w"] > 0 and n_edge > 0:
                ne = min(cfg["dag_edges"], n_edge)
                sel = torch.randint(0, n_edge, (ne,), device=DEV)
                loss = loss + cfg["dag_w"] * dag_hinge(logits, ec[sel], ep[sel], cfg["dag_margin"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        avg = tot / max(nb, 1); losses.append(avg)
        print(f"{log_prefix}seed{seed} ep{ep_i} loss {avg:.4f}", flush=True)
    return model, losses


# ----------------------------- eval -----------------------------
@torch.no_grad()
def recall_at_k(data, models, eval_idx, k=100, bs=512):
    """Mean top-k recall over eval proteins for an ensemble (averaged scores)."""
    GO_T = data.GO_V.t().contiguous()
    rec = []; capped = []
    for s in range(0, len(eval_idx), bs):
        bidx = eval_idx[s:s + bs]
        x = data.codes[bidx]
        scores = None
        for m in models:
            m.eval()
            lg = m(x, GO_T)
            scores = lg if scores is None else scores + lg
        topk = scores.topk(k, dim=1).indices.cpu().numpy()
        for b, i in enumerate(bidx):
            pos = set(data.positives(i).tolist())
            if not pos:
                continue
            hit = len(pos & set(topk[b].tolist()))
            rec.append(hit / len(pos))
            capped.append(min(k, len(pos)) / len(pos))
    return float(np.mean(rec)), float(np.mean(capped)), len(rec)


def recall_tf_baseline(data, eval_idx, k=100):
    """Predict the k globally most frequent vocab terms for everyone."""
    top = set(np.argsort(-data.tf)[:k].tolist())
    rec = []
    for i in eval_idx:
        pos = set(data.positives(i).tolist())
        if not pos:
            continue
        rec.append(len(pos & top) / len(pos))
    return float(np.mean(rec))
