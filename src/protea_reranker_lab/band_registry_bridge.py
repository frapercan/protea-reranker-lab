"""Band-registry bridge: resolve (band, cutoff) -> (OBO path, IA path).

This module vendors a snapshot of the BANDS constants from
``protea.core.band_registry`` (origin/develop, commit read 2026-06-07) so
the lab does not need a live PROTEA import.  The bridge adds a
host-specific IA-file resolver that maps the canonical ``ia_tokens`` of a
band to actual file paths on disk, via an ordered search list that never
falls back to the old hardcoded
``/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv``.

Usage::

    from protea_reranker_lab.band_registry_bridge import resolve_band_artifacts

    obo_path, ia_path = resolve_band_artifacts("v227")

A cross-band IA/snapshot mix raises :class:`BandMismatchError` (re-exported
from this module).

Design: the BANDS dict mirrors ``protea.core.band_registry.BANDS``
verbatim; do NOT alter it without also updating the upstream registry.  The
file-resolver walks ``IA_SEARCH_ROOTS`` in order and returns the first path
whose basename (case-insensitive) matches any of the band's ``ia_tokens``.
``OBO_SEARCH_ROOTS`` is the analogous list for the ontology file.

Environment overrides (the recommended escape hatch for CI environments or
custom dataset layouts; they override the search-list entirely for that band):

* ``LAB_IA_<BAND>``  e.g. ``LAB_IA_V227=/path/to/IA.tsv`` (uppercase)
* ``LAB_OBO_<BAND>``  e.g. ``LAB_OBO_V227=/path/to/go.obo`` (uppercase)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

__all__ = [
    "Band",
    "BANDS",
    "BandMismatchError",
    "resolve_band",
    "resolve_band_artifacts",
    "ia_token",
    "band_for_ia_token",
    "band_for_obo_version",
    "assert_band_consistency",
]


class BandMismatchError(ValueError):
    """A cell's snapshot or IA artifact belongs to a band other than the
    one it declares (the phantom-gap guard fired)."""


@dataclass(frozen=True)
class Band:
    """Canonical (ontology snapshot, IA) binding for one evaluation window."""

    name: str
    obo_versions: frozenset[str] = field(default_factory=frozenset)
    ia_tokens: frozenset[str] = field(default_factory=frozenset)
    description: str = ""
    t0_cutoff: date | None = None

    def accepts_obo_version(self, obo_version: str | None) -> bool:
        return obo_version is not None and obo_version in self.obo_versions

    def accepts_ia_token(self, token: str | None) -> bool:
        if token is None:
            return False
        return token.lower() in {t.lower() for t in self.ia_tokens}


# Vendored snapshot of PROTEA's band_registry.BANDS (origin/develop 2026-06-07).
# Update both here and upstream whenever a new GOA band is added.
BANDS: dict[str, Band] = {
    "v226": Band(
        name="v226",
        obo_versions=frozenset({"releases/2025-03-16"}),
        ia_tokens=frozenset({"IA_cafa6.tsv"}),
        t0_cutoff=date(2025, 5, 3),
        description=(
            "Historical benchmark cut (GOA v226, t0 2025-05-03). "
            "Congruent GO ontology = releases/2025-03-16. "
            "IA = IA_cafa6.tsv."
        ),
    ),
    "v227": Band(
        name="v227",
        obo_versions=frozenset({"releases/2025-07-22"}),
        ia_tokens=frozenset({"IA.tsv", "IA-swissprot-exp-v227.txt"}),
        t0_cutoff=date(2025, 9, 4),
        description=(
            "Deployed LAFA window (GOA v227 to v230, t0 2025-09-04). "
            "Congruent GO ontology = releases/2025-07-22. "
            "Canonical IA = lafa_t0_Sep_2025/IA.tsv (39906 terms). "
            "IA_cafa6.tsv is REJECTED for this band."
        ),
    ),
}

# Ordered search roots for IA files.  The resolver walks each root (one level)
# looking for a file whose basename matches a band's ia_tokens.
# Roots are checked in order; the first match wins.
_BASE = Path("/home/frapercan/Thesis2")
IA_SEARCH_ROOTS: list[Path] = [
    _BASE / "protea-lafa-knn" / "lafa_t0_Sep_2025",
    _BASE / "repositories" / "protea-reranker-lab" / "datasets" / "bench-v1-K5",
    _BASE / "storage",
    _BASE / "tmp-546" / "data" / "benchmarks",
    _BASE / "snapshots" / "PROTEA" / "data" / "benchmarks",
]

# Ordered search roots for OBO files.
OBO_SEARCH_ROOTS: list[Path] = [
    _BASE / "protea-lafa-knn" / "lafa_t0_Sep_2025",
    _BASE / "repositories" / "protea-reranker-lab" / "datasets" / "bench-v1-K5",
    _BASE / "storage",
]


def ia_token(ia_ref: str | None) -> str | None:
    """Stable IA-artifact token from a path or URL (basename, no query)."""
    if not ia_ref:
        return None
    cleaned = re.split(r"[?#]", ia_ref, maxsplit=1)[0].rstrip("/")
    base = os.path.basename(cleaned)
    return base or None


def resolve_band(cutoff: str) -> Band:
    """Resolve a declared band/cutoff string to its canonical ``Band``.

    Accepts the band name directly (``"v226"``) or a ``vNNN``-bearing token
    (``"bench-v1-K5-v226-lineage-prostt5"``) and extracts the band. Raises
    ``BandMismatchError`` for an unknown band.
    """
    if cutoff in BANDS:
        return BANDS[cutoff]
    for token in re.findall(r"v\d+", cutoff or ""):
        if token in BANDS:
            return BANDS[token]
    known = ", ".join(sorted(BANDS))
    raise BandMismatchError(
        f"Unknown band/cutoff {cutoff!r}; registered bands are: {known}. "
        "Add a Band row to band_registry_bridge.BANDS to register it."
    )


def band_for_obo_version(obo_version: str | None) -> str | None:
    """Reverse lookup: band name whose canonical snapshot universe contains
    ``obo_version``, or ``None``."""
    for band in BANDS.values():
        if band.accepts_obo_version(obo_version):
            return band.name
    return None


def band_for_ia_token(token: str | None) -> str | None:
    """Reverse lookup: band name whose canonical IA tokens contain ``token``,
    or ``None``."""
    for band in BANDS.values():
        if band.accepts_ia_token(token):
            return band.name
    return None


def assert_band_consistency(
    declared_band: str,
    *,
    obo_version: str | None,
    ia_ref: str | None,
) -> Band:
    """Reject a cell whose snapshot or IA come from a foreign band.

    Raises ``BandMismatchError`` when obo_version or ia_ref do not belong
    to the declared band.  Returns the resolved ``Band`` on success.
    """
    band = resolve_band(declared_band)

    if not band.accepts_obo_version(obo_version):
        other = band_for_obo_version(obo_version)
        suffix = f" (it belongs to band {other!r})" if other else ""
        raise BandMismatchError(
            f"Phantom-gap guard: cell declares band {band.name!r} but its "
            f"pivot ontology snapshot obo_version {obo_version!r} is not "
            f"canonical for that band{suffix}. Canonical obo_versions for "
            f"{band.name!r}: {sorted(band.obo_versions)}."
        )

    token = ia_token(ia_ref)
    if token is None:
        raise BandMismatchError(
            f"Phantom-gap guard: cell declares band {band.name!r} but no IA "
            f"artifact resolved. A band-declared cell must never fall back to "
            f"uniform IC=1. Pin the canonical IA for {band.name!r}: "
            f"{sorted(band.ia_tokens)}."
        )
    if not band.accepts_ia_token(token):
        other = band_for_ia_token(token)
        suffix = f" (it belongs to band {other!r})" if other else ""
        raise BandMismatchError(
            f"Phantom-gap guard: cell declares band {band.name!r} but its IA "
            f"artifact {token!r} is not canonical for that band{suffix}. "
            f"Canonical IA tokens for {band.name!r}: {sorted(band.ia_tokens)}."
        )
    return band


def _find_ia_file(band: Band) -> Path | None:
    """Walk IA_SEARCH_ROOTS and return the first file matching a band ia_token."""
    expected_tokens = {t.lower() for t in band.ia_tokens}
    for root in IA_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_file() and candidate.name.lower() in expected_tokens:
                return candidate
    return None


def _obo_version(path: Path) -> str | None:
    """Read the ``data-version:`` header from a GO OBO file (first 50 lines)."""
    try:
        with path.open() as fh:
            for i, line in enumerate(fh):
                if i >= 50:
                    break
                if line.startswith("data-version:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _find_obo_file(band: Band) -> Path | None:
    """Walk OBO_SEARCH_ROOTS and return the first .obo file whose data-version
    belongs to the declared band.

    The version is read from the OBO header (``data-version:`` line) and
    matched against ``band.obo_versions``.  Returns ``None`` when no
    band-congruent file is found; the caller should then require
    ``LAB_OBO_<BAND>`` to be set.
    """
    for root in OBO_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if not (candidate.is_file() and candidate.suffix == ".obo"):
                continue
            version = _obo_version(candidate)
            if version is not None and band.accepts_obo_version(version):
                return candidate
    return None


def _resolve_ia_path(
    band: Band, env_prefix: str, ia_path: Path | str | None
) -> Path:
    """Resolve the IA file path for ``band``, then validate its token.

    Precedence: explicit ``ia_path`` keyword > ``LAB_IA_<BAND>`` env var >
    search-list discovery.  Raises ``FileNotFoundError`` when the file is
    absent and ``BandMismatchError`` when the token is cross-band.
    """
    ia: Path
    if ia_path is not None:
        ia = Path(ia_path)
    else:
        env_ia = os.environ.get(f"LAB_IA_{env_prefix}")
        if env_ia:
            ia = Path(env_ia)
        else:
            found_ia = _find_ia_file(band)
            if found_ia is None:
                raise FileNotFoundError(
                    f"No IA file found for band {band.name!r}. "
                    f"Expected tokens: {sorted(band.ia_tokens)}. "
                    f"Searched: {[str(r) for r in IA_SEARCH_ROOTS]}. "
                    f"Set LAB_IA_{env_prefix}=/path/to/IA.tsv to override."
                )
            ia = found_ia

    if not ia.exists():
        raise FileNotFoundError(
            f"IA file {ia} does not exist (resolved for band {band.name!r})."
        )
    token = ia_token(str(ia))
    if not band.accepts_ia_token(token):
        other = band_for_ia_token(token)
        suffix = f" (it belongs to band {other!r})" if other else ""
        raise BandMismatchError(
            f"IA file {ia.name!r} is not canonical for band {band.name!r}"
            f"{suffix}. Canonical IA tokens: {sorted(band.ia_tokens)}."
        )
    return ia


def _resolve_obo_path(
    band: Band, env_prefix: str, obo_path: Path | str | None
) -> Path:
    """Resolve the OBO file path for ``band``.

    Precedence: explicit ``obo_path`` keyword > ``LAB_OBO_<BAND>`` env var >
    search-list discovery.  Raises ``FileNotFoundError`` when the file is
    absent.
    """
    obo: Path
    if obo_path is not None:
        obo = Path(obo_path)
    else:
        env_obo = os.environ.get(f"LAB_OBO_{env_prefix}")
        if env_obo:
            obo = Path(env_obo)
        else:
            found_obo = _find_obo_file(band)
            if found_obo is None:
                raise FileNotFoundError(
                    f"No OBO file found for band {band.name!r}. "
                    f"Searched: {[str(r) for r in OBO_SEARCH_ROOTS]}. "
                    f"Set LAB_OBO_{env_prefix}=/path/to/go.obo to override."
                )
            obo = found_obo

    if not obo.exists():
        raise FileNotFoundError(
            f"OBO file {obo} does not exist (resolved for band {band.name!r})."
        )
    return obo


def resolve_band_artifacts(
    band_or_cutoff: str,
    *,
    obo_path: Path | str | None = None,
    ia_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Resolve (band_or_cutoff) -> (obo_path, ia_path).

    The caller may supply explicit overrides via ``obo_path`` / ``ia_path``
    (e.g. from CLI flags or test fixtures).  Environment variables
    ``LAB_OBO_<BAND>`` and ``LAB_IA_<BAND>`` (band name uppercased, hyphens
    to underscores) take precedence over search-list discovery but are
    overridden by the explicit keyword arguments.

    A ``BandMismatchError`` is raised when:

    - the band is unknown,
    - the resolved OBO version does not belong to the declared band (if the
      obo_path carries a version header), or
    - the resolved IA token does not belong to the declared band.

    The function NEVER returns a path that does not exist on disk.
    """
    band = resolve_band(band_or_cutoff)
    env_prefix = band.name.upper().replace("-", "_")
    ia = _resolve_ia_path(band, env_prefix, ia_path)
    obo = _resolve_obo_path(band, env_prefix, obo_path)
    return obo, ia
