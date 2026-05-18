"""Unit tests for the contracts shim (LP.2).

Verifies that KnnContext and FeatureBuildContext are reachable from the
lab's public contract surface and behave correctly as frozen parameter
objects sourced from protea-contracts>=0.2.0.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from protea_reranker_lab.contracts import FeatureBuildContext, KnnContext


class TestKnnContextShim:
    def test_importable_from_contracts(self) -> None:
        assert KnnContext is not None

    def test_instantiation_required_fields_only(self) -> None:
        emb = np.zeros((2, 128), dtype=np.float32)
        ctx = KnnContext(
            query_accessions=["P12345", "Q67890"],
            query_embeddings=emb,
        )
        assert ctx.query_accessions == ["P12345", "Q67890"]
        assert ctx.query_embeddings is emb

    def test_optional_fields_default_none(self) -> None:
        emb = np.zeros((1, 64), dtype=np.float32)
        ctx = KnnContext(query_accessions=["P00001"], query_embeddings=emb)
        assert ctx.query_sequences is None
        assert ctx.query_tax_ids is None
        assert ctx.query_known_gos is None
        assert ctx.ref_data is None
        assert ctx.ref_sequences is None
        assert ctx.ref_tax_ids is None
        assert ctx.neighbors is None

    def test_frozen(self) -> None:
        emb = np.zeros((1, 64), dtype=np.float32)
        ctx = KnnContext(query_accessions=["P00001"], query_embeddings=emb)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ctx.query_accessions = ["Q00001"]  # type: ignore[misc]

    def test_full_fields(self) -> None:
        emb = np.zeros((2, 128), dtype=np.float32)
        ctx = KnnContext(
            query_accessions=["P12345", "Q67890"],
            query_embeddings=emb,
            query_sequences={"P12345": "MAFLR", "Q67890": "MGKLT"},
            query_tax_ids={"P12345": 9606, "Q67890": None},
            query_known_gos={"P12345": {"GO:0001234"}},
            ref_data={"BPO": {"P99999": {}}},
            ref_sequences={"P99999": "MAAAA"},
            ref_tax_ids={"P99999": 10090},
            neighbors=[[(("P99999"), 0.12)]],
        )
        assert ctx.query_sequences == {"P12345": "MAFLR", "Q67890": "MGKLT"}
        assert ctx.query_tax_ids["P12345"] == 9606


class TestFeatureBuildContextShim:
    def test_importable_from_contracts(self) -> None:
        assert FeatureBuildContext is not None

    def test_instantiation_all_defaults(self) -> None:
        fbc = FeatureBuildContext()
        assert fbc.ia_weights is None
        assert fbc.pca_state is None
        assert fbc.embedding_pool is None
        assert fbc.pivot_go_ids is None
        assert fbc.parent_map is None
        assert fbc.go_id_map is None
        assert fbc.aspect_map is None
        assert fbc.gt_pairs is None

    def test_frozen(self) -> None:
        fbc = FeatureBuildContext(ia_weights={"GO:0001234": 0.5})
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            fbc.ia_weights = {}  # type: ignore[misc]

    def test_ia_weights_field(self) -> None:
        weights = {"GO:0001234": 0.5, "GO:0009999": 1.2}
        fbc = FeatureBuildContext(ia_weights=weights)
        assert fbc.ia_weights == weights

    def test_pca_state_field(self) -> None:
        mean = np.zeros(128, dtype=np.float32)
        components = np.eye(16, 128, dtype=np.float32)
        fbc = FeatureBuildContext(pca_state=(mean, components))
        assert fbc.pca_state is not None
        assert fbc.pca_state[0] is mean
        assert fbc.pca_state[1] is components

    def test_gt_pairs_field(self) -> None:
        pairs: frozenset[tuple[str, str]] = frozenset({("P12345", "GO:0001234")})
        fbc = FeatureBuildContext(gt_pairs=pairs)
        assert ("P12345", "GO:0001234") in fbc.gt_pairs


class TestShimReexport:
    """Verify context types are also reachable from the lab package root."""

    def test_top_level_import(self) -> None:
        from protea_reranker_lab import FeatureBuildContext as FBC
        from protea_reranker_lab import KnnContext as KC
        assert KC is KnnContext
        assert FBC is FeatureBuildContext

    def test_same_class_as_protea_contracts(self) -> None:
        from protea_contracts.contexts import FeatureBuildContext as _FBC
        from protea_contracts.contexts import KnnContext as _KC
        assert KnnContext is _KC
        assert FeatureBuildContext is _FBC
