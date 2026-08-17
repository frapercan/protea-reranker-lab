"""Resolve the host locations this lab needs, from the environment or not at all.

Every location below used to be an absolute path under one developer's home
directory, written into a default argument. That worked until the machine was
reinstalled under a different user name, at which point the paths pointed at
nothing. Nothing raised: the ground truth directory was passed to a subprocess
that failed, the failure was caught and turned into an error dictionary, the
metric became ``None``, and the run wrote ``status: ok`` with every cell empty.

So the defect was never the user name. It was that a location the run cannot
work without was expressed as a default rather than as a requirement, and a
default cannot fail loudly.

Each resolver here takes an environment variable first, falls back to a location
derived from this file's own position in the checkout, and raises
:class:`MissingHostPath` naming the variable when neither exists. Raising is the
point. A lab that stops with "set PROTEA_LAB_GT_DIR" costs a minute; a lab that
runs to completion reporting nothing costs a day and a wrong conclusion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "MissingHostPath",
    "thesis_root",
    "ground_truth_dir",
    "protea_python",
    "ia_table",
]


class MissingHostPath(RuntimeError):
    """A location the run cannot proceed without is absent.

    Carries the environment variable to set, because the caller is a technician
    who wants the fix rather than the diagnosis.
    """

    def __init__(self, what: str, env_var: str, tried: list[Path]) -> None:
        attempts = "\n  ".join(str(p) for p in tried)
        super().__init__(
            f"{what} not found. Set {env_var}, or place it at one of:\n  {attempts}"
        )
        self.env_var = env_var
        self.tried = tried


def _resolve(what: str, env_var: str, candidates: list[Path], *, must_be_file: bool) -> Path:
    """First existing candidate, environment variable ahead of every fallback."""
    tried: list[Path] = []
    override = os.environ.get(env_var)
    if override:
        candidates = [Path(override).expanduser(), *candidates]
    for candidate in candidates:
        tried.append(candidate)
        if candidate.is_file() if must_be_file else candidate.is_dir():
            return candidate
    raise MissingHostPath(what, env_var, tried)


def thesis_root() -> Path:
    """The working root that holds ``repositories/``, ``CAFA_forever/`` and the rest.

    Derived from this file's location rather than from a home directory, so a
    checkout that moves keeps working and a checkout on another machine does not
    need editing. ``src/protea_reranker_lab/host_paths.py`` sits four levels
    below the repository, and the repository sits under ``repositories/``.
    """
    override = os.environ.get("THESIS_ROOT")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[4]


def ground_truth_dir() -> Path:
    """Directory of the evaluation release holding ``groundtruth_targets.tsv`` and friends."""
    return _resolve(
        "the evaluation ground-truth directory",
        "PROTEA_LAB_GT_DIR",
        [thesis_root() / "CAFA_forever" / "data" / "releases" / "Sep_2025_Mar_2026"],
        must_be_file=False,
    )


def protea_python() -> Path:
    """Interpreter that can import ``cafaeval``.

    Falls back to the interpreter running this process, which is correct
    whenever the lab and the scorer share an environment and is a great deal
    better than a path into a virtualenv that may not exist.
    """
    return _resolve(
        "a Python interpreter with cafaeval installed",
        "PROTEA_LAB_PYTHON",
        [thesis_root() / "repositories" / "PROTEA" / ".venv" / "bin" / "python",
         Path(sys.executable)],
        must_be_file=True,
    )


def ia_table() -> Path:
    """Information-accretion table used to weight the scorer."""
    return _resolve(
        "the information-accretion table",
        "PROTEA_LAB_IA_PATH",
        [thesis_root() / "protea-lafa-knn" / "lafa_t0_Sep_2025" / "IA.tsv"],
        must_be_file=True,
    )
