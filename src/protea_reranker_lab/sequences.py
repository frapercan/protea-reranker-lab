"""Lazy parquet feature loaders for LightGBM.

A :class:`ParquetFeatureSequence` is an :class:`lgb.Sequence` that exposes
the feature matrix of one or more sorted parquet files as if it were a single
dense float32 array, without ever holding more than one row group in RAM.

LightGBM constructs its internal binned dataset by iterating sequentially
through the sequence(s); after binning it can free the raw data
(``free_raw_data=True``). Random-access reads (used by bagging / row sampling)
are supported but optimised for sorted index batches via row-group caching.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq


class ParquetFeatureSequence(lgb.Sequence):
    """Lazy float32 feature matrix backed by one or more parquet files.

    Files are concatenated in the given order; each file is expected to be
    self-contained (rows sorted by ``protein_accession`` within the file, and
    no protein crosses file boundaries — this is the invariant produced by
    :mod:`staging`).
    """

    batch_size = 65_536  # LightGBM reads __getitem__ in chunks of this size

    def __init__(self, parquet_paths: list[str | Path], feature_cols: list[str]):
        if not parquet_paths:
            raise ValueError("ParquetFeatureSequence requires at least one parquet file")
        self._files = [pq.ParquetFile(str(p)) for p in parquet_paths]
        self._cols = list(feature_cols)

        rg_table: list[tuple[int, int, int, int, int]] = []  # (file_idx, rg_idx, start, stop, n)
        offset = 0
        for fi, pf in enumerate(self._files):
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                rg_table.append((fi, rg, offset, offset + n, n))
                offset += n
        self._rg_table = rg_table
        self._n = offset
        self._rg_starts = np.fromiter((r[2] for r in rg_table), dtype=np.int64, count=len(rg_table))

        self._cache_key: tuple[int, int] | None = None
        self._cache_arr: np.ndarray | None = None

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._n)
            if step != 1:
                raise NotImplementedError("non-unit slice steps are not supported")
            return self._read_range(start, stop)
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            if i < 0:
                i += self._n
            return self._read_range(i, i + 1)[0]
        arr = np.asarray(idx)
        if arr.dtype == bool:
            return self._read_range(0, self._n)[arr]
        return self._read_indices(arr.astype(np.int64, copy=False))

    def _load_rg(self, file_idx: int, rg_idx: int) -> np.ndarray:
        key = (file_idx, rg_idx)
        if self._cache_key == key and self._cache_arr is not None:
            return self._cache_arr
        tab = self._files[file_idx].read_row_group(rg_idx, columns=self._cols)
        cols = []
        for name in self._cols:
            col = tab.column(name).to_numpy(zero_copy_only=False)
            if col.dtype != np.float64:
                col = col.astype(np.float64, copy=False)
            cols.append(col)
        out = np.column_stack(cols).astype(np.float64, copy=False)
        self._cache_key = key
        self._cache_arr = out
        return out

    def _read_range(self, start: int, stop: int) -> np.ndarray:
        if start >= stop:
            return np.empty((0, len(self._cols)), dtype=np.float64)
        first_rg = int(np.searchsorted(self._rg_starts, start, side="right") - 1)
        chunks: list[np.ndarray] = []
        for rg_idx in range(first_rg, len(self._rg_table)):
            file_idx, rg, rg_start, rg_stop, _ = self._rg_table[rg_idx]
            if rg_stop <= start:
                continue
            if rg_start >= stop:
                break
            arr = self._load_rg(file_idx, rg)
            local_start = max(0, start - rg_start)
            local_stop = min(rg_stop - rg_start, stop - rg_start)
            chunks.append(arr[local_start:local_stop])
        if not chunks:
            return np.empty((0, len(self._cols)), dtype=np.float64)
        if len(chunks) == 1:
            return np.ascontiguousarray(chunks[0])
        return np.concatenate(chunks, axis=0)

    def _read_indices(self, indices: np.ndarray) -> np.ndarray:
        n = len(indices)
        if n == 0:
            return np.empty((0, len(self._cols)), dtype=np.float64)
        order = np.argsort(indices, kind="stable")
        sorted_idx = indices[order]
        out = np.empty((n, len(self._cols)), dtype=np.float64)
        rg_for_row = np.searchsorted(self._rg_starts, sorted_idx, side="right") - 1
        i = 0
        while i < n:
            rg = int(rg_for_row[i])
            j = i + 1
            while j < n and rg_for_row[j] == rg:
                j += 1
            file_idx, rg_idx, rg_start, _, _ = self._rg_table[rg]
            arr = self._load_rg(file_idx, rg_idx)
            local = sorted_idx[i:j] - rg_start
            out[order[i:j]] = arr[local]
            i = j
        return out
