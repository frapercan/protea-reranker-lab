"""Filter a frozen reranker parquet dataset to remove cross-category replicas.

Background. PROTEA's ``train_reranker.py`` writes the same ``(protein,
candidate)`` row into all three category buckets (``nk``, ``lk``, ``pk``)
with ``label`` re-assigned per category. The lab then trains a per-category
reranker by filtering ``category == X``. Result: within ``category=nk``,
positives only come from genuinely-NK proteins (count==0) while negatives
come from a mix of NK genuines and contamination from LK/PK proteins
(count>0). The model picks up ``count == 0 ⇒ positive`` and inflates Fmax.

This script materialises a corrected parquet pair where each category bucket
only contains proteins genuinely categorised as that category, identified
empirically as ``(protein, aspect)`` tuples having at least one ``label=1``
row in that category bucket.

Pass 1 — scan and collect ``{cat: {(protein, aspect)}}`` tuples with
positives. Pass 2 — stream the file again applying the membership filter,
write a new parquet next to the original.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def collect_genuine_keys(
    path: Path,
) -> dict[str, set[tuple[str, str, str]]]:
    """Return ``{category: {(snapshot_pair, protein, aspect)}}`` for tuples
    with positives. Per-snapshot filtering matters because a protein can be
    NK in an early snapshot pair (no annotations at t0) and LK/PK in a
    later one (gained context). Only the snapshot where the protein
    genuinely belongs to the category should survive."""
    dataset = ds.dataset(str(path), format="parquet")
    keys: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    scanner = dataset.scanner(
        columns=["protein_accession", "aspect", "category", "snapshot_pair", "label"],
        filter=ds.field("label") == 1,
        batch_size=500_000,
        use_threads=True,
    )
    n = 0
    for batch in scanner.to_batches():
        prots = batch["protein_accession"].to_numpy(zero_copy_only=False)
        asps = batch["aspect"].to_numpy(zero_copy_only=False)
        cats = batch["category"].to_numpy(zero_copy_only=False)
        snaps = batch["snapshot_pair"].to_numpy(zero_copy_only=False)
        for prot, asp, cat, snap in zip(prots, asps, cats, snaps):
            keys[str(cat)].add((str(snap), str(prot), str(asp)))
        n += len(prots)
    print(f"  scanned {n:,} positive rows; categories: "
          + ", ".join(f"{c}={len(v):,}" for c, v in keys.items()))
    return keys


def filter_into(
    src: Path,
    dst: Path,
    genuine: dict[str, set[tuple[str, str, str]]],
) -> dict[str, int]:
    """Stream ``src`` → ``dst`` keeping only rows whose
    (cat, snapshot_pair, prot, aspect) is in ``genuine``."""
    dataset = ds.dataset(str(src), format="parquet")
    schema = dataset.schema
    writer = pq.ParquetWriter(str(dst), schema, compression="snappy")
    stats: dict[str, int] = defaultdict(int)
    for batch in dataset.to_batches(batch_size=500_000, use_threads=True):
        prots = batch["protein_accession"].to_numpy(zero_copy_only=False)
        asps = batch["aspect"].to_numpy(zero_copy_only=False)
        cats = batch["category"].to_numpy(zero_copy_only=False)
        snaps = batch["snapshot_pair"].to_numpy(zero_copy_only=False)
        mask = pa.array([
            (str(snap), str(prot), str(asp)) in genuine.get(str(cat), set())
            for prot, asp, cat, snap in zip(prots, asps, cats, snaps)
        ], type=pa.bool_())
        kept = batch.filter(mask)
        if kept.num_rows:
            writer.write_batch(kept)
            for cat in kept["category"].to_numpy(zero_copy_only=False):
                stats[str(cat)] += 1
        stats["_seen"] += batch.num_rows
        if stats["_seen"] % 10_000_000 < 500_000:
            print(f"    seen {stats['_seen']:,}, kept "
                  + ", ".join(f"{c}={stats[c]:,}" for c in ("nk","lk","pk")))
    writer.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("datasets/bench-v1-K5"))
    ap.add_argument("--dst", type=Path,
                    default=Path("datasets/bench-v1-K5-filtered"))
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"== Pass 1: collect genuine (prot, aspect) per category from {src.name}/train.parquet")
    train_keys = collect_genuine_keys(src / "train.parquet")
    print(f"== Pass 1: same for {src.name}/eval.parquet")
    eval_keys = collect_genuine_keys(src / "eval.parquet")

    print(f"\n== Pass 2: filter train → {dst}/train.parquet")
    train_stats = filter_into(src / "train.parquet", dst / "train.parquet", train_keys)
    print(f"  train kept: {dict(train_stats)}")

    print(f"== Pass 2: filter eval → {dst}/eval.parquet")
    eval_stats = filter_into(src / "eval.parquet", dst / "eval.parquet", eval_keys)
    print(f"  eval kept: {dict(eval_stats)}")

    # Copy auxiliary files (manifest, go.obo, parent_map.json) and amend manifest name.
    import shutil
    for aux in ("go.obo", "parent_map.json"):
        if (src / aux).exists():
            shutil.copy2(src / aux, dst / aux)

    manifest = json.loads((src / "manifest.json").read_text())
    manifest["name"] = manifest.get("name", src.name) + "-filtered"
    manifest["filter_provenance"] = {
        "source_dataset": src.name,
        "rule": (
            "Keep rows where (category, protein_accession, aspect) has at "
            "least one label=1 row in that category. Removes cross-category "
            "replicas introduced by parquet_export.py."
        ),
        "train_rows_kept_per_cat": {k: train_stats.get(k, 0) for k in ("nk","lk","pk")},
        "eval_rows_kept_per_cat": {k: eval_stats.get(k, 0) for k in ("nk","lk","pk")},
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {dst / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
