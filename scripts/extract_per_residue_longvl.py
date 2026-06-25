#!/usr/bin/env python
"""Extract per-residue ankh-base states for a BOUNDED long / very-long sample.

The shipped per-residue sample (``per_residue_v227/``) was a flat 5000-protein draw, so
its long bucket is thin (N~142) and its very-long bucket is unusable (N=3). The factorial
substrate study needs a real N in those buckets to compare residue + learned aggregation
against the champion. This script tops up the SAME directory with per-residue arrays for a
bounded sample of long (970..1831) and very-long (>1831) v227-annotated proteins.

GPU forward passes (ankh-base), shortest-first, per-protein incremental save, OOM-safe. The
output is transient: only the float16 npy arrays are stored (one per accession), exactly as
the original extractor did. Read-only on the DB (sequences are pulled by the TSV builder, not
here). Run in the deploy/GPU env with ~11.5GB free.

Usage::

    python scripts/extract_per_residue_longvl.py \
        --tsv  /home/frapercan/Thesis2/storage/fullgo_models/per_residue_longvl_sample.tsv \
        --out  /home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # annotations only; torch is a heavy GPU dep, imported lazily in main()
    import torch as _torch  # noqa: F401

MODEL = "ElnaggarLab/ankh-base"
BACKENDS_SRC = "/home/frapercan/Thesis2/repositories/protea-backends/src"


def _noop_emit(*_a, **_k):
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True, help="accession<TAB>sequence rows to extract")
    ap.add_argument("--out", required=True, help="output dir for <accession>.npy float16 arrays")
    ap.add_argument("--max-residues", type=int, default=3000,
                    help="skip sequences longer than this (12GB GPU guard)")
    args = ap.parse_args()

    import torch

    sys.path.insert(0, BACKENDS_SRC)
    from protea_backends.ankh import AnkhBackend

    t0 = time.time()
    accs, seqs = [], []
    with open(args.tsv) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1] and len(parts[1]) <= args.max_residues:
                accs.append(parts[0])
                seqs.append(parts[1])
    if not accs:
        print("no eligible sequences in TSV", flush=True)
        return 1
    print(f"sampled {len(accs)} proteins (len {min(map(len, seqs))}-{max(map(len, seqs))})",
          flush=True)

    be = AnkhBackend()
    model, tok = be.load_model(MODEL, "cuda", emit=_noop_emit)
    print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))  # shortest first
    done = set(os.listdir(args.out))
    ok = skip = 0
    for n, i in enumerate(order):
        acc, seq = accs[i], seqs[i]
        if f"{acc}.npy" in done:
            ok += 1
            continue
        try:
            try:
                tensors = be._compute_residue_tensors(model, tok, [seq], layers=[0],
                                                      layer_agg="mean")
            except TypeError:
                tensors = be._compute_residue_tensors(model, tok, [seq], layers=[0],
                                                      layer_agg="mean", emit=_noop_emit)
            t = tensors[0]
            arr = (t.detach().float().cpu().numpy() if hasattr(t, "detach")
                   else np.asarray(t)).astype(np.float16)
            np.save(os.path.join(args.out, f"{acc}.npy"), arr)
            ok += 1
        except torch.cuda.OutOfMemoryError:
            skip += 1
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            skip += 1
            print(f"  skip {acc} (len {len(seq)}): {type(exc).__name__}", flush=True)
        if n % 100 == 0:
            torch.cuda.empty_cache()
            print(f"  {n}/{len(seqs)} ok={ok} skip={skip} ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE: {ok} saved to {args.out}/, {skip} skipped ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
