"""Whether a dictionary can serve as a retrieval index, and how to tell.

The served encoder is trained so code cosine matches functional similarity, and
nothing in that objective asks its atoms to be discriminative. On the shipped bank
the consequence is one atom of 2048 in half of 575,503 rows and a top-10 posting
head carrying 0.69 to 1.11 nats against a median of 3.18.

The measurements here exist because a previous reading of the same bank concluded
sparse retrieval was closed. It measured CANDIDATE FRACTION, which saturates as
soon as one atom is shared and therefore cannot see a reduction in work at all.
Measuring postings touched, and recall of the true neighbours inside a shortlist
rather than exact agreement under a pruned query, reverses the conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from protea_reranker_lab.dictionary_usage import (
    RetrievalProbe,
    effective_width,
    posting_lengths,
    two_stage_recall,
)


def _uniform_bank(n=400, k=8, d=64, seed=0):
    """Every atom equally likely: the dictionary a retrieval index wants."""
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(d, size=k, replace=False) for _ in range(n)]).astype(np.int32)
    val = rng.random((n, k)).astype(np.float32)
    return idx, val


def _collapsed_bank(n=400, k=8, d=64, seed=0):
    """Half the code is one hot atom, which is the shape the shipped bank has."""
    rng = np.random.default_rng(seed)
    idx, val = _uniform_bank(n, k, d, seed)
    idx[:, 0] = 0
    val[:, 0] = 5.0
    return idx, val


# ------------------------------------------------------------------ effective width

def test_a_uniform_dictionary_uses_all_of_itself():
    idx, _ = _uniform_bank(d=64)

    got = effective_width(idx, 64)

    assert got["used_fraction"] > 0.9
    assert got["dead_atoms"] == 0


def test_a_collapsed_dictionary_is_narrower_than_its_label():
    """A 2048 dictionary at exp(H) 548 is a 548-atom dictionary wearing a larger
    label, and collision computed on the nominal width understates overlap."""
    uniform, _ = _uniform_bank(d=64)
    collapsed, _ = _collapsed_bank(d=64)

    assert effective_width(collapsed, 64)["effective_width"] < \
           effective_width(uniform, 64)["effective_width"]


def test_an_empty_code_stops_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="no atoms"):
        effective_width(np.zeros((0, 8), dtype=np.int32), 64)


def test_dead_atoms_are_counted():
    idx, _ = _uniform_bank(d=128)

    assert effective_width(idx, 512)["dead_atoms"] >= 384


# --------------------------------------------------------------------- the index head

def test_the_head_is_what_decides_whether_pruning_is_possible():
    """An atom in half the bank puts half the bank in every candidate set that
    touches it, whatever the rest of the code does."""
    idx, _ = _collapsed_bank()

    got = posting_lengths(idx, 64)

    assert got["longest_share"] == pytest.approx(1.0)
    assert got["atoms_over_half"] >= 1


def test_a_uniform_bank_has_no_head():
    idx, _ = _uniform_bank(n=400, k=8, d=64)

    assert posting_lengths(idx, 64)["atoms_over_half"] == 0


def test_the_head_carries_less_information_than_the_median():
    idx, _ = _collapsed_bank()

    got = posting_lengths(idx, 64)

    assert got["top10_idf"][0] < got["median_idf"]


# ------------------------------------------------------------------- two-stage recall

def test_keeping_every_atom_recovers_the_truth():
    idx, val = _uniform_bank(n=300, k=8, d=64)
    qs = np.arange(20)

    got = two_stage_recall(idx, val, 64, RetrievalProbe(queries=qs, keep=8, shortlist=50))

    assert got["recall"] == pytest.approx(1.0)
    assert got["speedup"] == pytest.approx(1.0, abs=0.01)


def test_pruning_the_query_cuts_work():
    """The measure a candidate fraction cannot see: postings touched."""
    idx, val = _collapsed_bank(n=300, k=8, d=64)
    qs = np.arange(20)

    full = two_stage_recall(idx, val, 64, RetrievalProbe(queries=qs, keep=8, shortlist=50))
    pruned = two_stage_recall(idx, val, 64, RetrievalProbe(queries=qs, keep=3, shortlist=50))

    assert pruned["postings"] < full["postings"]
    assert pruned["speedup"] > 1.0


def test_a_wider_shortlist_recovers_more_of_the_truth():
    """Exact agreement under a pruned query measures a system nobody would build;
    a second exact pass over the shortlist recovers the order."""
    idx, val = _uniform_bank(n=400, k=8, d=64)
    qs = np.arange(25)

    narrow = two_stage_recall(idx, val, 64, RetrievalProbe(queries=qs, keep=2, shortlist=20))
    wide = two_stage_recall(idx, val, 64, RetrievalProbe(queries=qs, keep=2, shortlist=200))

    assert wide["recall"] >= narrow["recall"]


def test_the_result_carries_the_probe_that_produced_it():
    idx, val = _uniform_bank(n=200, k=8, d=64)

    got = two_stage_recall(idx, val, 64, RetrievalProbe(queries=np.arange(10), keep=4))

    assert got["keep"] == 4 and got["queries"] == 10 and got["shortlist"] == 1000
