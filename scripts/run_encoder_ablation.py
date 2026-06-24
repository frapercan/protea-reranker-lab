"""Thin entry for the learned-encoder ablation study (logic in src/encoder_ablation.py).

Usage:
    poetry run python scripts/run_encoder_ablation.py                 # esm2_150m defaults
    poetry run python scripts/run_encoder_ablation.py --ref-n 100000 --epochs 200
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 poetry run python scripts/run_encoder_ablation.py
"""
from __future__ import annotations

import argparse
import logging

from protea_reranker_lab.encoder_ablation import EncoderAblationSpec, run_encoder_ablation


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [encoder-ablation] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=EncoderAblationSpec.name)
    ap.add_argument("--embedding-config-id", default=EncoderAblationSpec.embedding_config_id)
    ap.add_argument("--band", default=EncoderAblationSpec.band)
    ap.add_argument("--ref-n", type=int, default=EncoderAblationSpec.ref_n)
    ap.add_argument("--knn", type=int, default=EncoderAblationSpec.knn)
    ap.add_argument("--epochs", type=int, default=EncoderAblationSpec.epochs)
    ap.add_argument("--train-pairs", type=int, default=EncoderAblationSpec.train_pairs)
    ap.add_argument("--seed", type=int, default=EncoderAblationSpec.seed)
    ap.add_argument("--official", action="store_true",
                    help="score with the official LAFA harness (toi + PK-known exclusion)")
    args = ap.parse_args()

    base = EncoderAblationSpec()
    toi = base.gt_dir / "groundtruth_terms_of_interest.txt"
    pk_known = base.gt_dir / "groundtruth_PK_known.tsv"
    spec = EncoderAblationSpec(
        name=args.name, embedding_config_id=args.embedding_config_id, band=args.band,
        ref_n=args.ref_n, knn=args.knn, epochs=args.epochs, train_pairs=args.train_pairs,
        seed=args.seed, official_harness=args.official,
        toi_path=toi if args.official else None,
        pk_known_path=pk_known if args.official else None,
    )
    report = run_encoder_ablation(spec)
    print(f"\nDONE {report['name']} (hash {report['spec_hash']}):")
    for name, r in report["results"].items():
        print(f"  {name:22s} NK+LK={r['nklk_mean']} PK={r['pk_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
