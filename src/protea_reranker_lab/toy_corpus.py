"""A corpus small enough to run every arm on, and honest enough to catch a broken one.

The pieces of this study now span extraction, normalisation, one or two learned
projections, aggregation, sparsification, quantisation and retrieval. Composing
them for the first time on real data means a null result has two explanations, the
hypothesis and the plumbing, and no way to separate them.

So the plumbing is settled here first, on a corpus whose answer is known by
construction. Proteins belong to families; a family owns a set of motifs and a set
of GO terms; a residue is a noisy draw from its family's motifs. An encoding that
works recovers the family structure, and neighbour purity has a ceiling that can
be computed rather than guessed.

WHAT THE TOY IS FOR AND WHAT IT IS NOT

It is for: does the pipeline run, does each stage change what it claims to change,
does a deliberately broken arm actually score worse, and where does trainability
break as the dictionary widens. That last one is the question the bit-budget
arithmetic cannot answer, since it says a 65,536-atom dictionary is affordable and
says nothing about whether 60,000 proteins can populate it.

It is NOT for: any number that goes in a chapter. The families here are cleanly
separable in a way real function is not, so every arm scores higher than it will
on protein data, and the ORDER is the only thing worth carrying forward, and only
as a hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToySpec:
    """Knobs that decide how hard the toy is.

    ``family_overlap`` is the one that matters: at 0 the families share no terms
    and any encoding separates them, which flatters everything. Real function
    overlaps heavily, so a toy at 0 would rank arms by how well they solve a
    problem nobody has.
    """

    proteins: int = 400
    families: int = 48
    dim: int = 64
    #: Motifs are drawn from ONE SHARED POOL and a family is defined by which
    #: subset it uses, not by owning a centre of its own. With private motifs the
    #: family means come out nearly orthogonal in 64 dimensions and plain mean
    #: pooling recovers everything, which is how the first version of this toy
    #: scored every arm at exactly the ceiling and ordered nothing.
    motif_pool: int = 20
    motifs_per_family: int = 6
    #: Residues carrying the family's marker motif, as an ABSOLUTE COUNT rather
    #: than a share. This is the whole reason the toy dilutes: a functional domain
    #: is a fixed number of residues, so it occupies a tenth of a 1,000-residue
    #: protein and a thirtieth of a 3,000-residue one, and mean pooling loses it
    #: progressively with length.
    #:
    #: Parameterised as a fraction first, which was wrong and the toy caught it:
    #: a constant fraction gives a long protein MORE marked residues and better
    #: noise averaging, so long proteins scored best and the dilution signature
    #: came out inverted.
    marker_length: int = 4
    #: Markers come from a shared pool and a family is the SUBSET it carries, so
    #: detecting one marker never identifies a family and the combination has to be
    #: read. With a private marker per family the problem is a one-of-N detection,
    #: which a sparse code solves outright and which saturated the toy at 100 per
    #: cent for every arm that could see a single atom.
    marker_pool: int = 8
    markers_per_family: int = 5
    marker_strength: float = 1.0
    terms_per_family: int = 8
    family_overlap: float = 0.35
    noise: float = 3.0
    length_min: int = 40
    length_max: int = 600
    seed: int = 42


@dataclass(frozen=True)
class ToyCorpus:
    """Residues, closures and the family each protein came from."""

    residues: dict[str, np.ndarray]
    closures: dict[str, frozenset[str]]
    family: dict[str, int]
    spec: ToySpec

    @property
    def accessions(self) -> list[str]:
        return sorted(self.residues)

    def lengths(self) -> dict[str, int]:
        return {a: int(r.shape[0]) for a, r in self.residues.items()}


def build_toy(spec: ToySpec | None = None) -> ToyCorpus:
    """Generate the corpus. Deterministic under the seed, families balanced."""
    s = spec or ToySpec()
    rng = np.random.default_rng(s.seed)

    # One pool, subsets per family. Two families sharing most of their motifs have
    # close means and differ mainly in COMPOSITION, which is what a mean cannot see
    # and a code over the right atoms can.
    pool_motifs = rng.standard_normal((s.motif_pool, s.dim)).astype(np.float32)
    marker_bank = (rng.standard_normal((s.marker_pool, s.dim)).astype(np.float32)
                   * s.marker_strength)
    family_markers = np.stack([
        rng.choice(s.marker_pool, size=s.markers_per_family, replace=False)
        for _ in range(s.families)
    ])

    # Terms are drawn from a shared pool so families overlap the way function does.
    pool = [f"GO:{i:05d}" for i in range(int(s.families * s.terms_per_family * 0.7))]
    family_terms = []
    for f in range(s.families):
        own = rng.choice(len(pool), size=s.terms_per_family, replace=False)
        family_terms.append({pool[i] for i in own})
    # Force the requested overlap by sharing a slice of family 0's terms outward.
    shared = sorted(family_terms[0])[: max(1, int(s.terms_per_family * s.family_overlap))]
    for f in range(1, s.families):
        family_terms[f] |= set(shared)

    residues, closures, family = {}, {}, {}
    for i in range(s.proteins):
        f = i % s.families
        length = int(rng.integers(s.length_min, s.length_max + 1))
        # Background from the shared pool, then a marker in a minority of positions.
        which = rng.integers(0, s.motif_pool, size=length)
        x = pool_motifs[which].copy()
        n_marked = min(length, s.marker_length)
        marked = rng.choice(length, size=n_marked, replace=False)
        # The family's markers are spread across the marked positions, so no single
        # position identifies the family and the combination carries the signal.
        assigned = family_markers[f][rng.integers(0, s.markers_per_family, size=n_marked)]
        x[marked] = marker_bank[assigned]
        x = x + s.noise * rng.standard_normal((length, s.dim)).astype(np.float32)
        accession = f"T{i:05d}"
        residues[accession] = x.astype(np.float32)
        closures[accession] = frozenset(family_terms[f] | {f"GO:own-{i}"})
        family[accession] = f
    return ToyCorpus(residues, closures, family, s)


def ceiling_purity(corpus: ToyCorpus, k: int = 10) -> float:
    """The ORACLE: the k most functionally similar proteins that exist, per protein.

    The first version took an arbitrary k from the same family, which is not a
    ceiling: an encoder that finds the BEST same-family neighbours beats it, and
    arms duly reported above 100 per cent of range, which is how the mistake
    surfaced. The oracle is what no encoder can exceed, so a fraction of it is a
    fraction of what was achievable.
    """
    from protea_reranker_lab.functional_proxy import jaccard

    accessions = corpus.accessions
    scores = []
    for a in accessions:
        sims = sorted(
            (jaccard(corpus.closures[a], corpus.closures[b]) for b in accessions if b != a),
            reverse=True,
        )[:k]
        if sims:
            scores.append(float(np.mean(sims)))
    return float(np.mean(scores)) if scores else 0.0


def floor_purity(corpus: ToyCorpus, k: int = 10, seed: int = 0) -> float:
    """Purity a random encoder reaches, which is the other end of the scale.

    An arm that beats nothing is worth knowing about immediately, and on a toy
    with overlapping families the random floor is well above zero.
    """
    from protea_reranker_lab.functional_proxy import jaccard

    rng = np.random.default_rng(seed)
    accessions = corpus.accessions
    scores = []
    for a in accessions:
        others = [b for b in accessions if b != a]
        picked = rng.choice(len(others), size=min(k, len(others)), replace=False)
        scores.append(float(np.mean([jaccard(corpus.closures[a], corpus.closures[others[i]])
                                     for i in picked])))
    return float(np.mean(scores))
