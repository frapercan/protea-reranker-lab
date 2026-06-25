"""Unit tests for the torch-free helpers of the backbone size-vs-signal sweep.

The GPU head training (champion ``fit_encoder`` recipe) and the DB pulls are exercised
by the script under GPU + a live Postgres; here we lock the pure helpers (length
bucketing, the readable MLflow metric naming, the truncation-clean ceiling) so a rename
or an off-by-one in the bucket edges is caught in CI without the heavy stack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_backbone_size_sweep",
    Path(__file__).resolve().parents[1] / "scripts" / "run_backbone_size_sweep.py",
)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
# Register before exec so the @dataclass in the module can resolve its own __module__
# (dataclasses look the class up in sys.modules during processing).
sys.modules["run_backbone_size_sweep"] = mod
_SPEC.loader.exec_module(mod)


def test_bucket_edges_are_contiguous_and_capped() -> None:
    # short/medium/long edges as documented; nothing below 0 or above MAX_LEN.
    assert mod.bucket_of(0) == "short"
    assert mod.bucket_of(318) == "short"
    assert mod.bucket_of(319) == "medium"
    assert mod.bucket_of(969) == "medium"
    assert mod.bucket_of(970) == "long"
    assert mod.bucket_of(mod.MAX_LEN) == "long"


def test_bucket_out_of_range_is_none() -> None:
    # past the truncation-clean ceiling there is no bucket (the protein is dropped).
    assert mod.bucket_of(mod.MAX_LEN + 1) is None


def test_metric_name_is_readable() -> None:
    assert mod._metric_name("long", "learned", "resnik") == "spearman_learned_resnik_long"
    assert mod._metric_name("short", "dense-mean-cosine", "lin") == "spearman_dense_lin_short"


def test_small_esm2_end_is_in_the_backbone_set() -> None:
    # The decisive read needs the few-million-param ESM2 end present in the sweep.
    keys = {b.key for b in mod.BACKBONES}
    assert {"esm2-8m", "esm2-150m", "esm2-650m", "esm2-3b"} <= keys


def test_parse_vec_roundtrip() -> None:
    v = mod._parse_vec("[1.0,-2.5,3.0]")
    assert list(v) == [1.0, -2.5, 3.0]
