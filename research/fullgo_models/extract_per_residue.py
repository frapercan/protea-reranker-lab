"""Extract PER-RESIDUE ankh-base embeddings for a v227 sample (SDR grid Phase 2, A2).

The platform has no pooling="none" config, so for the experiment we extract per-residue
states directly via the AnkhBackend (correct ankh tokenisation, is_split_into_words). Saves a
concatenated float16 array + offsets + accessions, so the A2 scripts can sparsify-each-residue-
then-bundle WITHOUT any mean-pooling collapse. Read-only DB. GPU (deploy env).
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/protea-backends/src")
from protea_backends.ankh import AnkhBackend  # noqa: E402

MODEL = "ElnaggarLab/ankh-base"
TSV = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_sample.tsv"
OUTDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"


def noop_emit(*a, **k):
    return None


def main():
    t0 = time.time()
    accs, seqs = [], []
    for ln in open(TSV):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) == 2 and parts[1]:
            accs.append(parts[0]); seqs.append(parts[1])
    print(f"sampled {len(accs)} proteins (len {min(map(len,seqs))}-{max(map(len,seqs))})", flush=True)

    be = AnkhBackend()
    model, tok = be.load_model(MODEL, "cuda", emit=noop_emit)
    print(f"model loaded ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    # shortest first: robust if a few long ones still OOM; incremental per-protein save.
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    done = set(os.listdir(OUTDIR))
    ok = skip = 0
    for n, i in enumerate(order):
        acc, seq = accs[i], seqs[i]
        if f"{acc}.npy" in done:
            ok += 1; continue
        try:
            try:
                tensors = be._compute_residue_tensors(model, tok, [seq], layers=[0], layer_agg="mean")
            except TypeError:
                tensors = be._compute_residue_tensors(model, tok, [seq], layers=[0],
                                                      layer_agg="mean", emit=noop_emit)
            t = tensors[0]
            arr = (t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)).astype(np.float16)
            np.save(os.path.join(OUTDIR, f"{acc}.npy"), arr)
            ok += 1
        except torch.cuda.OutOfMemoryError:
            skip += 1
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            skip += 1
            print(f"  skip {acc} (len {len(seq)}): {type(e).__name__}", flush=True)
        if n % 200 == 0:
            torch.cuda.empty_cache()
            print(f"  {n}/{len(seqs)} ok={ok} skip={skip} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE: {ok} proteins saved to {OUTDIR}/, {skip} skipped ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
