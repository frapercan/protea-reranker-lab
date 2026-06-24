"""Train + persist ONE production learned encoder (logic in src/encoder_ablation.py).

Produces the artifact a downstream apply step (PROTEA op) loads to project any protein's
mean-pooled embedding into the learned GO-aligned code. Defaults to prot_t5 + hard-neg (the
ablation winner).

Usage:
    poetry run python scripts/train_learned_encoder.py \\
        --embedding-config-id 084943c6-fec1-441d-bdc5-63b0268ada1b \\
        --ref-n 150000 --epochs 200 --out storage/learned_encoders/prot_t5_hardneg.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from protea_reranker_lab.encoder_ablation import ArmSpec, EncoderAblationSpec, train_and_save_encoder


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [train-encoder] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-config-id", default="084943c6-fec1-441d-bdc5-63b0268ada1b")
    ap.add_argument("--band", default="v227")
    ap.add_argument("--ref-n", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--train-pairs", type=int, default=400000)
    ap.add_argument("--dict-dim", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--objective", default="hard-neg", choices=["cosine-lin", "hard-neg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="artifact path (.pt)")
    args = ap.parse_args()

    spec = EncoderAblationSpec(
        name=f"prod-encoder-{args.objective}", embedding_config_id=args.embedding_config_id,
        band=args.band, ref_n=args.ref_n, epochs=args.epochs, train_pairs=args.train_pairs,
        seed=args.seed,
    )
    arm = ArmSpec(name=f"learned-{args.objective}", kind="learned", dict_dim=args.dict_dim,
                  top_k=args.top_k, objective=args.objective)
    meta = train_and_save_encoder(spec, arm, Path(args.out))
    print(f"\nDONE: encoder saved to {args.out}\n  meta={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
