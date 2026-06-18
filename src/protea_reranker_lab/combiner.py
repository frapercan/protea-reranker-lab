"""Combiner mode (MR-2): the shallow per-category meta-learner over the score vector.

The meta-reranker (see ``agent-farm/plans/meta-reranker/ARCHITECTURE.md``) splits
the old monolithic 73-feature booster into a WIDE set of independent evidence
scorers (ports) feeding a SHALLOW per-category combiner. This module is the lab
side of slice MR-2: it trains a LightGBM combiner over ONLY the small score
vector (a handful of component scores), per category, instead of the full feature
matrix. The monolith path in :mod:`protea_reranker_lab.train` is untouched; the
combiner is purely additive and opt-in via ``--combiner``.

Score-vector column contract
----------------------------
The columns the combiner consumes are the per-candidate signals the PROTEA
EvidenceScorer adapters read (PROTEA #645,
``protea/core/reranking/scorers.py``). Each adapter reads ONE stored key off a
candidate; the combiner consumes the same raw stored keys (they ride the
``GOPrediction.features`` JSONB per #643 and land in the export parquet, e.g.
dataset ``fullgo-native-parity-SELECT-220-227-assocfix``). Keeping this list in
sync with the adapters is the single coupling point.

=============== ============================================ ===================
scorer (port)   stored column(s)                              applies to
=============== ============================================ ===================
alignment       ``alignment_score_sw``                        NK / LK / PK
taxonomy        ``taxonomic_distance``                        NK / LK / PK
label_embedding ``anc2vec_neighbor_maxcos``                   NK / LK / PK
knn_similarity  ``distance`` + ``neighbor_vote_fraction``     NK / LK / PK
classifier      ``classifier_score``                          NK / LK / PK
self_prior      ``self_prior_score``                          LK / PK only
association     ``association_total`` + ``association_cross``  LK / PK only
=============== ============================================ ===================

The first three rows are the BASE sequence / taxonomy / label-embedding evidence
(MR-2): they are NOT priors, so they apply to every category and are what gives
the NK cells real signal beyond knn + classifier. Only the last two scorers are
priors keyed on the query's own pre-cutoff known terms, so only they are dropped
for NK.

``go_term_frequency`` (term_frequency) and ``interpro_score`` (interpro) were
dropped from the vector (MR-2.5): on the 7401 LAFA frame the base-enriched
11-column vector scored MEAN 0.259 vs MEAN 0.310 for this evidence-only vector,
because the IA-weighted f_micro_w rewards RARE informative terms while
``go_term_frequency`` biases toward FREQUENT low-IA terms, and ``interpro_score``
is an export-only column (0 percent importance everywhere and not computable at
PROTEA predict time).

For NK the two priors are absent / zero (the query has no pre-cutoff known
terms), so :func:`combiner_columns_for_category` drops them for ``nk`` cells. The
columns are raw stored signals, NOT the adapters' calibrated transforms: the
adapter calibration (e.g. ``1 - distance`` times the vote fraction) is the
scorer's responsibility at PROTEA inference time; the combiner learns its own
weighting over whatever raw signals the parquet carries, which is exactly the
"combiner consumes scores, never raw features" invariant at the level the lab
sees (these handful of columns ARE the score vector relative to the 73-feature
matrix).
"""

from __future__ import annotations

#: Categories with pre-cutoff known terms (the two priors are undefined on NK).
#: Mirrors ``_KNOWN_CATEGORIES`` in PROTEA ``protea/core/reranking/scorers.py``.
_KNOWN_CATEGORIES: frozenset[str] = frozenset({"lk", "pk"})

#: Stored columns each scorer port reads. The key is the scorer ``name`` used in
#: PROTEA's ``default_scorer_registry``; the value is the ordered list of raw
#: parquet columns that feed it. The combiner's input vector is the concatenation
#: of these in registration order (base evidence first, then knn_similarity,
#: classifier, self_prior, association) so the column order is stable + auditable.
#: The three base-evidence scorers (alignment, taxonomy, label_embedding) are NOT
#: priors: they apply to every category and restore the sequence / taxonomy /
#: label-embedding signal the NK cells lost to the priors-only vector. The
#: IA-harmful ``term_frequency`` + export-only ``interpro`` scorers were dropped
#: (MR-2.5); see the module docstring.
SCORE_VECTOR_BY_SCORER: dict[str, list[str]] = {
    "alignment": ["alignment_score_sw"],
    "taxonomy": ["taxonomic_distance"],
    "label_embedding": ["anc2vec_neighbor_maxcos"],
    "knn_similarity": ["distance", "neighbor_vote_fraction"],
    "classifier": ["classifier_score"],
    "self_prior": ["self_prior_score"],
    "association": ["association_total", "association_cross"],
}

#: Scorers whose evidence only exists for LK / PK proteins (priors need the
#: query's own pre-cutoff known terms, which NK proteins lack by definition).
_KNOWN_ONLY_SCORERS: frozenset[str] = frozenset({"self_prior", "association"})

#: The full, ordered score-vector column list (all categories, before the NK
#: prior-drop). Registration order = canonical combiner input order.
DEFAULT_COMBINER_COLUMNS: list[str] = [
    col for scorer in SCORE_VECTOR_BY_SCORER for col in SCORE_VECTOR_BY_SCORER[scorer]
]


def combiner_columns_for_category(category: str) -> list[str]:
    """Return the ordered score-vector columns the combiner consumes for ``category``.

    ``category`` is the cell prefix (``nk`` / ``lk`` / ``pk``, case-insensitive).
    NK drops the two prior scorers (self_prior, association) because their stored
    signals are absent / zero for proteins with no pre-cutoff known terms, exactly
    as ``applies_to`` declares in PROTEA's scorer adapters. LK / PK get the full
    vector.
    """
    cat = category.lower()
    cols: list[str] = []
    for scorer, scorer_cols in SCORE_VECTOR_BY_SCORER.items():
        if scorer in _KNOWN_ONLY_SCORERS and cat not in _KNOWN_CATEGORIES:
            continue
        cols.extend(scorer_cols)
    return cols


def resolve_combiner_columns(cell: str, explicit: list[str] | None = None) -> list[str]:
    """Resolve the combiner feature columns for ``cell`` (e.g. ``pk-bpo``).

    When ``explicit`` is given (operator-supplied ``--combiner-columns``) it wins
    verbatim, in the given order, with no per-category drop (the operator owns the
    contract). Otherwise the default per-category score vector is used, with the
    NK prior-drop applied by :func:`combiner_columns_for_category`.
    """
    if explicit:
        # De-duplicate while preserving the operator's order.
        seen: set[str] = set()
        out: list[str] = []
        for col in explicit:
            if col not in seen:
                seen.add(col)
                out.append(col)
        return out
    category = cell.lower().split("-", 1)[0]
    return combiner_columns_for_category(category)


__all__ = [
    "DEFAULT_COMBINER_COLUMNS",
    "SCORE_VECTOR_BY_SCORER",
    "combiner_columns_for_category",
    "resolve_combiner_columns",
]
