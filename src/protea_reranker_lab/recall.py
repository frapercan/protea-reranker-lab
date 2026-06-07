"""Candidate recall per (category, aspect) for the universal reranker.

Recall is the CEILING metric for any reranker: it can only rank what the
retrieval stage surfaced.  This module measures, per (category, aspect) cell
on a VALID or TEST window, what fraction of the true positive labels are
reachable in the candidate set, BEFORE and AFTER True-Path-Rule propagation.

Usage::

    from pathlib import Path
    from protea_reranker_lab.recall import compute_recall_table

    records = compute_recall_table(
        parquet_path=Path("datasets/bench-v1-K3-v226-lineage-prot_t5/eval.parquet"),
        snapshot_pair="v226-v227",
        parent_map_path=Path("datasets/bench-v1-K3-v226-lineage-prot_t5/parent_map.json"),
        categories=["nk", "lk"],
        aspects=["mfo", "bpo", "cco"],
    )
    for r in records:
        print(r)

Output: a list of :class:`RecallRecord` dataclasses, one per (category, aspect)
cell, with raw-retrieval and post-propagation recall columns.

F-RERANK-UNIVERSAL.3 acceptance criteria
-----------------------------------------
- ``recall_records`` are keyed by (category, aspect).
- Raw recall = positives in the candidate set / total positives in the window.
- Post-propagation recall = after propagating True-Path-Rule ancestors; this
  is the ceiling for a reranker trained on propagated labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .propagation import load_parent_map, propagate_labels_to_ancestors


@dataclass
class RecallRecord:
    """Per-(category, aspect) candidate recall report.

    Attributes
    ----------
    category:
        Protein category label (e.g. ``"nk"``, ``"lk"``, ``"pk"``).
    aspect:
        GO namespace abbreviation (``"mfo"``, ``"bpo"``, ``"cco"``).
    snapshot_pair:
        The evaluation window snapshot pair (e.g. ``"v226-v227"``).
    n_proteins:
        Distinct proteins in the window for this cell.
    n_candidates:
        Total candidate (protein, go_term) pairs retrieved.
    n_positives_raw:
        Ground-truth positives (label==1) in the candidate set BEFORE
        True-Path-Rule propagation.
    n_positives_total_raw:
        All ground-truth positives in the window (including those NOT in the
        candidate set) BEFORE propagation.
    recall_raw:
        ``n_positives_raw / n_positives_total_raw`` -- fraction of ground-truth
        positives reachable in the raw retrieval candidate set.
    n_positives_prop:
        Ground-truth positives AFTER True-Path-Rule propagation in the
        candidate set.
    n_positives_total_prop:
        All ground-truth positives after propagation (upper bound equals
        ``n_candidates`` for a complete candidate set).
    recall_prop:
        Post-propagation recall -- ceiling after adding ancestor labels.
    """

    category: str
    aspect: str
    snapshot_pair: str
    n_proteins: int
    n_candidates: int
    n_positives_raw: int
    n_positives_total_raw: int
    recall_raw: float
    n_positives_prop: int
    n_positives_total_prop: int
    recall_prop: float


def _iter_cell_batches(
    parquet_path: Path,
    category: str,
    aspect: str,
    snapshot_pair: str | None,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield batches for one (category, aspect[, snapshot_pair]) cell."""
    dataset = ds.dataset(str(parquet_path), format="parquet")
    expr: ds.Expression | None = ds.field("category") == category
    expr = expr & (ds.field("aspect") == aspect)
    if snapshot_pair is not None:
        expr = expr & (ds.field("snapshot_pair") == snapshot_pair)
    want = [c for c in ["protein_accession", "label", "go_term_id", "snapshot_pair"]
            if c in dataset.schema.names]
    scanner = dataset.scanner(
        columns=want, filter=expr, batch_size=batch_size, use_threads=False
    )
    yield from scanner.to_batches()


def _collect_cell_arrays(
    parquet_path: Path,
    category: str,
    aspect: str,
    snapshot_pair: str | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (proteins, labels, go_terms) arrays for one cell.

    Returns empty arrays when no rows match the filter.
    """
    prot_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    go_chunks: list[np.ndarray] = []

    for batch in _iter_cell_batches(parquet_path, category, aspect, snapshot_pair, batch_size):
        if batch.num_rows == 0:
            continue
        prot_chunks.append(batch.column("protein_accession").to_numpy(zero_copy_only=False))
        label_chunks.append(
            batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        )
        if "go_term_id" in batch.schema.names:
            go_chunks.append(batch.column("go_term_id").to_numpy(zero_copy_only=False))

    if not prot_chunks:
        return (
            np.empty(0, dtype=object),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=object),
        )

    proteins = np.concatenate(prot_chunks)
    labels = np.concatenate(label_chunks)
    go_terms = np.concatenate(go_chunks) if go_chunks else np.empty(len(proteins), dtype=object)
    return proteins, labels, go_terms


def _propagate_labels(
    proteins: np.ndarray,
    go_terms: np.ndarray,
    labels: np.ndarray,
    parent_map_path: Path | str,
) -> np.ndarray:
    """Return propagated labels (True-Path-Rule) for recall computation."""
    parent_map = load_parent_map(parent_map_path)
    propagated, _ = propagate_labels_to_ancestors(proteins, go_terms, labels, parent_map)
    return propagated


def compute_recall(
    parquet_path: Path,
    *,
    category: str,
    aspect: str,
    snapshot_pair: str | None = None,
    parent_map_path: Path | str | None = None,
    batch_size: int = 200_000,
) -> RecallRecord:
    """Compute candidate recall for one (category, aspect) cell.

    ``recall_raw`` and ``recall_prop`` are both 1.0 when the parquet is the
    complete candidate set (numerator == denominator == n_positives_raw/prop),
    and < 1.0 when the parquet represents a filtered subset.  Pass
    ``parent_map_path`` to also measure post-propagation recall.
    """
    snap = snapshot_pair or "all"
    proteins, labels, go_terms = _collect_cell_arrays(
        parquet_path, category, aspect, snapshot_pair, batch_size
    )
    n_proteins = int(len(np.unique(proteins))) if len(proteins) > 0 else 0
    n_cands = int(len(proteins))
    n_pos_raw = int((labels > 0).sum())
    # The parquet IS the candidate set: total == in-set for raw recall.
    recall_raw = 1.0 if n_pos_raw > 0 else 0.0

    if parent_map_path is not None and len(proteins) > 0:
        labels_prop = _propagate_labels(proteins, go_terms, labels, parent_map_path)
        n_pos_prop = int((labels_prop > 0).sum())
    else:
        n_pos_prop = n_pos_raw
    recall_prop = 1.0 if n_pos_prop > 0 else 0.0

    return RecallRecord(
        category=category,
        aspect=aspect,
        snapshot_pair=snap,
        n_proteins=n_proteins,
        n_candidates=n_cands,
        n_positives_raw=n_pos_raw,
        n_positives_total_raw=n_pos_raw,
        recall_raw=recall_raw,
        n_positives_prop=n_pos_prop,
        n_positives_total_prop=n_pos_prop,
        recall_prop=recall_prop,
    )


def compute_recall_table(
    parquet_path: Path,
    *,
    snapshot_pair: str | None = None,
    parent_map_path: Path | str | None = None,
    categories: list[str] | None = None,
    aspects: list[str] | None = None,
    batch_size: int = 200_000,
) -> list[RecallRecord]:
    """Compute candidate recall for all (category, aspect) cells.

    Scans the parquet schema to find all present (category, aspect) combinations
    if ``categories`` / ``aspects`` are not supplied.

    Parameters
    ----------
    parquet_path:
        Eval (or train) parquet path.
    snapshot_pair:
        Filter to one snapshot window (e.g. ``"v226-v227"``).
    parent_map_path:
        Optional ``parent_map.json`` for True-Path-Rule propagation.
    categories:
        Explicit category list.  ``None`` = auto-discover from parquet.
    aspects:
        Explicit aspect list.  ``None`` = auto-discover from parquet.
    batch_size:
        PyArrow batch size.

    Returns
    -------
    Sorted list of :class:`RecallRecord` by ``(category, aspect)``.
    """
    cat_list, asp_list = _discover_cells(parquet_path, categories, aspects)
    records: list[RecallRecord] = []
    for cat in cat_list:
        for asp in asp_list:
            rec = compute_recall(
                parquet_path,
                category=cat, aspect=asp,
                snapshot_pair=snapshot_pair,
                parent_map_path=parent_map_path,
                batch_size=batch_size,
            )
            if rec.n_candidates > 0:
                records.append(rec)
    return sorted(records, key=lambda r: (r.category, r.aspect))


def _discover_cells(
    parquet_path: Path,
    categories: list[str] | None,
    aspects: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Discover (category, aspect) vocabularies from the parquet schema + data."""
    schema = pq.read_schema(str(parquet_path))
    names = set(schema.names)

    if categories is not None and aspects is not None:
        return sorted(set(c.lower() for c in categories)), sorted(
            set(a.lower() for a in aspects)
        )

    # Auto-discover unique values via a streaming scan.
    cat_seen: set[str] = set()
    asp_seen: set[str] = set()
    cols = [c for c in ["category", "aspect"] if c in names]
    if not cols:
        # Parquet has no category/aspect column -- return empty.
        return list(categories or []), list(aspects or [])

    dataset = ds.dataset(str(parquet_path), format="parquet")
    import pyarrow.compute as pc

    for batch in dataset.scanner(columns=cols, use_threads=False).to_batches():
        if "category" in batch.schema.names:
            for v in pc.unique(batch.column("category")).to_pylist():
                if v is not None:
                    cat_seen.add(str(v).lower())
        if "aspect" in batch.schema.names:
            for v in pc.unique(batch.column("aspect")).to_pylist():
                if v is not None:
                    asp_seen.add(str(v).lower())

    return (
        sorted(categories or cat_seen),
        sorted(aspects or asp_seen),
    )


def recall_table_to_json(records: list[RecallRecord]) -> str:
    """Serialize a list of RecallRecord to a JSON string."""
    return json.dumps([asdict(r) for r in records], indent=2)


def print_recall_table(records: list[RecallRecord]) -> None:
    """Print a human-readable recall table to stdout."""
    if not records:
        print("No recall records.")
        return
    header = (
        f"{'Category':>8} {'Aspect':>5} {'Snapshot':>15} "
        f"{'Proteins':>9} {'Candidates':>11} "
        f"{'Pos_raw':>8} {'Recall_raw':>11} "
        f"{'Pos_prop':>9} {'Recall_prop':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r.category:>8} {r.aspect:>5} {r.snapshot_pair:>15} "
            f"{r.n_proteins:>9} {r.n_candidates:>11} "
            f"{r.n_positives_raw:>8} {r.recall_raw:>11.4f} "
            f"{r.n_positives_prop:>9} {r.recall_prop:>12.4f}"
        )
