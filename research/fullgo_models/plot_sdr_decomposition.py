"""SDR decomposition ladder figure: isolates sparsity vs binarisation as the
source of the dense-vs-sparse gap for GO-function transfer (ankh-base, v227)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K = [32, 64, 128]
data = {
    "Resnik": {"A": 0.2315, "B": [0.2111, 0.2175, 0.2203], "C": [0.1300, 0.1052, 0.0934]},
    "Lin":    {"A": 0.1962, "B": [0.1798, 0.1817, 0.1832], "C": [0.0874, 0.0633, 0.0604]},
}

C_DENSE = "#1f3a5f"   # A
C_REAL  = "#2e8b57"   # B
C_BIN   = "#c0392b"   # C

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=False)
fig.suptitle(
    "Where does the dense-vs-sparse gap come from?  Sparsity is nearly free; binarisation is the killer.",
    fontsize=13, fontweight="bold", y=0.99,
)

for ax, metric in zip(axes, ("Resnik", "Lin")):
    d = data[metric]
    x = np.arange(len(K))
    ax.axhline(d["A"], color=C_DENSE, ls="--", lw=2,
               label=f"A · dense (full, real, cosine) = {d['A']:.3f}")
    ax.plot(x, d["B"], "-o", color=C_REAL, lw=2.5, ms=9,
            label="B · sparse REAL (top-k + magnitudes, cosine)")
    ax.plot(x, d["C"], "-s", color=C_BIN, lw=2.5, ms=9,
            label="C · sparse BINARY (top-k, 1/0, Tanimoto)")

    # annotate the two gaps at k=128 (rightmost)
    xi = len(K) - 1
    a, b, c = d["A"], d["B"][xi], d["C"][xi]
    ax.annotate("", xy=(xi, b), xytext=(xi, a),
                arrowprops=dict(arrowstyle="<->", color=C_REAL, lw=1.6))
    ax.text(xi - 0.07, (a + b) / 2, f"sparsity\n{b - a:+.3f}\n(free)",
            ha="right", va="center", fontsize=9, color=C_REAL, fontweight="bold")
    ax.annotate("", xy=(xi, c), xytext=(xi, b),
                arrowprops=dict(arrowstyle="<->", color=C_BIN, lw=1.6))
    ax.text(xi - 0.07, (b + c) / 2, f"binarisation\n{c - b:+.3f}\n(the killer)",
            ha="right", va="center", fontsize=9, color=C_BIN, fontweight="bold")

    for xi2, (bv, cv) in enumerate(zip(d["B"], d["C"])):
        ax.text(xi2, bv + 0.006, f"{bv:.3f}", ha="center", va="bottom", fontsize=8, color=C_REAL)
        ax.text(xi2, cv - 0.006, f"{cv:.3f}", ha="center", va="top", fontsize=8, color=C_BIN)

    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in K])
    ax.set_xlabel("active dimensions kept (of 768)")
    ax.set_ylabel("Spearman r  (representation similarity vs GO-semantic)")
    ax.set_title(f"GO ground truth: {metric}", fontsize=11)
    ax.set_ylim(0, 0.27)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

fig.text(0.5, 0.005,
         "ankh-base embeddings, v227 t0 leakage-clean pool, 5000 proteins / 200k pairs, seed 42.  "
         "k-WTA = keep top-k coordinates by magnitude.  At fixed k, binary-cosine and Tanimoto give identical Spearman.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out = "/home/frapercan/Thesis2/storage/fullgo_models/sdr_decomposition.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
