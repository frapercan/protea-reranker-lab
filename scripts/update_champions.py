#!/usr/bin/env python
"""Champion tracker for FARM-EXP.4.

Walks a directory of FARM-EXP.3 run-record JSON files, groups them by
``(eval_set, tier, aspect)``, and writes a deterministic
``champions.md`` table plus per-triple symlinks under
``runs/champion/<eval_set>/<tier>/<aspect>``.

Promotion rule (per slice spec):

    A non-champion run becomes the new champion only if its 95%
    percentile CI lower-bound on fmax exceeds the prior champion's
    mean fmax.

The candidate ``fmax_samples`` array is the percentile bootstrap
distribution emitted by ``bootstrap_cis.RunRecord`` (FARM-EXP.3); the
champion tracker reuses that loader byte-for-byte so the two scripts
share a parse path.

Range pinning: every entry carries ``axis.eval_set`` (the validation
range). This is the explicit fix for the v18-era champion record that
floated across ranges (postmortem
``project_v18_selective_rerank``). Champions are never
range-unknown; records without an ``eval_set`` axis are skipped with
a warning.

Symlink path policy: RELATIVE. The runner writes ``os.path.relpath``
output as the target so the symlink stays valid across worktrees and
clones.

CLI surface::

    python scripts/update_champions.py             # dry-run, prints planned diff
    python scripts/update_champions.py --apply     # writes champions.md + symlinks
    python scripts/update_champions.py --explain bench-v1-K5-filtered:nk:bpo
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# pyproject.toml puts scripts/ on the pythonpath; share the FARM-EXP.3 loader.
import bootstrap_cis as bc

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO / "runs"
DEFAULT_CHAMPIONS_FILE = REPO / "champions.md"
DEFAULT_SYMLINK_ROOT = REPO / "runs" / "champion"

# Re-export the loader so tests can assert identity (we are reusing,
# not re-implementing).
_LOAD_RUNS = bc.load_runs


# ----------------------------------------------------------- model


@dataclass(frozen=True)
class TriplKey:
    """``(eval_set, tier, aspect)`` cell coordinate."""

    eval_set: str
    tier: str
    aspect: str

    def as_str(self) -> str:
        return f"{self.eval_set}:{self.tier}:{self.aspect}"

    @classmethod
    def parse(cls, raw: str) -> "TriplKey":
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"expected EVAL_SET:TIER:ASPECT, got {raw!r}"
            )
        eval_set, tier, aspect = parts
        if not (eval_set and tier and aspect):
            raise ValueError(
                f"empty component in {raw!r}; need EVAL_SET:TIER:ASPECT"
            )
        return cls(eval_set=eval_set, tier=tier, aspect=aspect)


@dataclass(frozen=True)
class Decision:
    """One promotion/skip step in the champion walk for a triple."""

    candidate_run_id: str
    candidate_shortid: str | None
    candidate_mean: float
    candidate_ci_lo: float
    candidate_ci_hi: float
    prior_run_id: str | None
    prior_mean: float | None
    verdict: str  # "seeded" | "promoted" | "rejected"
    reason: str


@dataclass(frozen=True)
class ChampionRecord:
    """The current champion of one ``(eval_set, tier, aspect)`` triple."""

    triple: TriplKey
    record: bc.RunRecord
    mean: float
    ci_lo: float
    ci_hi: float
    source_path: Path  # the run.json file the record was loaded from


# ----------------------------------------------------------- helpers


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    """Return (2.5%, 97.5%) quantiles of a samples array."""
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _split_cell(cell: str | None) -> tuple[str, str] | None:
    """Parse a ``<tier>-<aspect>`` cell token; ``None`` on malformed."""
    if not cell or "-" not in cell:
        return None
    parts = cell.split("-", 1)
    if not (parts[0] and parts[1]):
        return None
    return parts[0], parts[1]


def _record_eval_set(record: bc.RunRecord) -> str | None:
    val = record.axis.get("eval_set")
    if val is None:
        return None
    return str(val)


def _record_chronology_key(record: bc.RunRecord, source: Path) -> tuple[Any, ...]:
    """Deterministic ordering key for the champion walk.

    Run IDs of the form ``YYYYMMDDTHHMMSS_*`` sort chronologically
    when compared as strings; we lean on that and fall back to the
    source path's mtime + name for records without timestamp
    prefixes.
    """
    rid = record.run_id or ""
    if rid[:8].isdigit() and len(rid) >= 15 and rid[8] == "T":
        return (0, rid)
    try:
        mtime = source.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (1, mtime, source.name)


def _load_records_with_source(
    runs_dir: Path, *, strict: bool = False
) -> list[tuple[bc.RunRecord, Path]]:
    """Variant of ``bc.load_runs`` that retains the source path.

    Walks ``runs_dir`` recursively for ``*.json`` files because the
    lab's runs tree is nested (``runs/<study>/<cell>/run.json``),
    unlike ``bc.load_runs`` which scans a flat directory of pre-staged
    bootstrap records. Records that fail the FARM-EXP.3 parse are
    skipped with a warning in non-strict mode.
    """
    import json

    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        # The lab gitignores runs/; the dir may legitimately be
        # absent in a fresh clone or worktree. Treat as empty.
        print(
            f"[info] runs dir not present: {runs_dir} "
            "(treating as empty)",
            file=sys.stderr,
        )
        return []
    if not runs_dir.is_dir():
        raise NotADirectoryError(
            f"runs path is not a directory: {runs_dir}"
        )
    out: list[tuple[bc.RunRecord, Path]] = []
    for path in sorted(runs_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise
            print(
                f"[warn] skipping {path}: invalid JSON ({exc})",
                file=sys.stderr,
            )
            continue
        if not isinstance(payload, dict):
            continue
        try:
            record = bc.RunRecord.from_dict(payload, source=str(path))
        except (ValueError, TypeError) as exc:
            if strict:
                raise
            # The lab has many legacy run.json files without
            # fmax_samples; we silently skip the noisiest case to
            # avoid spamming the operator.
            if "fmax_samples" not in str(exc):
                print(
                    f"[warn] skipping {path}: {exc}",
                    file=sys.stderr,
                )
            continue
        out.append((record, path))
    return out


# ----------------------------------------------------------- grouping


def group_by_triple(
    records: Iterable[tuple[bc.RunRecord, Path]],
) -> dict[TriplKey, list[tuple[bc.RunRecord, Path]]]:
    """Bucket records by their ``(eval_set, tier, aspect)`` coord.

    Records missing either an ``eval_set`` axis or a parseable
    ``cell`` token are dropped with a warning.
    """
    buckets: dict[TriplKey, list[tuple[bc.RunRecord, Path]]] = {}
    for record, source in records:
        eval_set = _record_eval_set(record)
        if eval_set is None:
            print(
                f"[warn] {source}: missing axis.eval_set; skipping "
                "(champions must be range-pinned)",
                file=sys.stderr,
            )
            continue
        cell_parts = _split_cell(record.cell)
        if cell_parts is None:
            print(
                f"[warn] {source}: malformed cell token "
                f"{record.cell!r}; skipping",
                file=sys.stderr,
            )
            continue
        tier, aspect = cell_parts
        key = TriplKey(eval_set=eval_set, tier=tier, aspect=aspect)
        buckets.setdefault(key, []).append((record, source))
    return buckets


# ----------------------------------------------------------- promotion walk


def _walk_triple(
    bucket: list[tuple[bc.RunRecord, Path]],
) -> tuple[ChampionRecord | None, list[Decision]]:
    """Walk one bucket chronologically, applying the promotion rule.

    Returns the final champion (or ``None`` if the bucket is empty)
    and the ordered list of decisions taken. Decisions are useful
    for ``--explain``.
    """
    ordered = sorted(
        bucket, key=lambda pair: _record_chronology_key(pair[0], pair[1])
    )
    champion: ChampionRecord | None = None
    trace: list[Decision] = []
    for record, source in ordered:
        ci_lo, ci_hi = _percentile_ci(record.fmax_samples)
        mean = float(record.fmax_samples.mean())
        if champion is None:
            trace.append(
                Decision(
                    candidate_run_id=record.run_id,
                    candidate_shortid=record.shortid,
                    candidate_mean=mean,
                    candidate_ci_lo=ci_lo,
                    candidate_ci_hi=ci_hi,
                    prior_run_id=None,
                    prior_mean=None,
                    verdict="seeded",
                    reason="first record for this triple",
                )
            )
            triple = TriplKey(
                eval_set=_record_eval_set(record) or "",
                tier=(_split_cell(record.cell) or ("", ""))[0],
                aspect=(_split_cell(record.cell) or ("", ""))[1],
            )
            champion = ChampionRecord(
                triple=triple,
                record=record,
                mean=mean,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                source_path=source,
            )
            continue
        # Existing champion. Promotion rule: candidate CI lower bound
        # must exceed the prior champion's mean fmax.
        if ci_lo > champion.mean:
            trace.append(
                Decision(
                    candidate_run_id=record.run_id,
                    candidate_shortid=record.shortid,
                    candidate_mean=mean,
                    candidate_ci_lo=ci_lo,
                    candidate_ci_hi=ci_hi,
                    prior_run_id=champion.record.run_id,
                    prior_mean=champion.mean,
                    verdict="promoted",
                    reason=(
                        f"candidate CI lo {ci_lo:.4f} > prior mean "
                        f"{champion.mean:.4f}"
                    ),
                )
            )
            champion = ChampionRecord(
                triple=champion.triple,
                record=record,
                mean=mean,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                source_path=source,
            )
        else:
            trace.append(
                Decision(
                    candidate_run_id=record.run_id,
                    candidate_shortid=record.shortid,
                    candidate_mean=mean,
                    candidate_ci_lo=ci_lo,
                    candidate_ci_hi=ci_hi,
                    prior_run_id=champion.record.run_id,
                    prior_mean=champion.mean,
                    verdict="rejected",
                    reason=(
                        f"candidate CI lo {ci_lo:.4f} <= prior mean "
                        f"{champion.mean:.4f}"
                    ),
                )
            )
    return champion, trace


def compute_champions(
    buckets: dict[TriplKey, list[tuple[bc.RunRecord, Path]]],
) -> tuple[
    dict[TriplKey, ChampionRecord], dict[TriplKey, list[Decision]]
]:
    """Apply the promotion walk to every triple bucket."""
    winners: dict[TriplKey, ChampionRecord] = {}
    traces: dict[TriplKey, list[Decision]] = {}
    for key, bucket in buckets.items():
        champion, trace = _walk_triple(bucket)
        traces[key] = trace
        if champion is not None:
            winners[key] = champion
    return winners, traces


# ----------------------------------------------------------- markdown


CHAMPIONS_HEADER = (
    "# Champions\n"
    "\n"
    "Auto-generated by `scripts/update_champions.py`. Do not edit by "
    "hand; regenerate with `python scripts/update_champions.py "
    "--apply`.\n"
    "\n"
    "Each `(eval_set, tier, aspect)` triple has at most one champion. "
    "A non-champion run becomes the new champion only if its 95% "
    "percentile CI lower-bound on fmax exceeds the prior champion's "
    "mean fmax (FARM-EXP.4 promotion rule). Every entry pins the "
    "validation range via `eval_set`; range-unknown entries are "
    "rejected upstream.\n"
    "\n"
)

# Marker pair delimiting a manually-curated appendix that survives
# ``--apply`` re-renders. The appendix records champions for which no
# FARM-EXP.3-format run record exists yet (pre-FARM-EXP.5 writer
# slices). Once the writer lands and the auto-walker promotes a real
# record for the same ``(eval_set, tier, aspect)`` triple, the manual
# entry can be retired by deletion. The markers are HTML comments so
# they render invisibly in GitHub-flavoured Markdown.
MANUAL_ENTRIES_BEGIN = "<!-- MANUAL_ENTRIES_BEGIN -->"
MANUAL_ENTRIES_END = "<!-- MANUAL_ENTRIES_END -->"


def _format_axis(axis: dict[str, Any]) -> str:
    """Render an axis dict as ``key=value, key=value, ...`` (sorted)."""
    parts = []
    for k in sorted(axis):
        v = axis[k]
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def extract_manual_appendix(existing_text: str) -> str:
    """Pull the manual-entries appendix out of an existing champions.md.

    Returns the substring delimited by :data:`MANUAL_ENTRIES_BEGIN`
    and :data:`MANUAL_ENTRIES_END` markers (markers included), with a
    leading newline so it appends cleanly. Returns an empty string
    when either marker is absent or the file has no manual entries.
    """
    begin = existing_text.find(MANUAL_ENTRIES_BEGIN)
    end = existing_text.find(MANUAL_ENTRIES_END)
    if begin == -1 or end == -1 or end < begin:
        return ""
    # Include the END marker itself.
    end_inclusive = end + len(MANUAL_ENTRIES_END)
    block = existing_text[begin:end_inclusive]
    if not block.endswith("\n"):
        block += "\n"
    return "\n" + block


def render_champions_md(
    winners: dict[TriplKey, ChampionRecord],
    *,
    manual_appendix: str = "",
) -> str:
    """Render the full champions.md body.

    ``manual_appendix`` is the preserved manual-entries block extracted
    from the existing file via :func:`extract_manual_appendix`. It is
    appended verbatim after the auto-generated body so manually
    documented champions (records with no FARM-EXP.3 run.json yet)
    survive ``--apply`` runs.
    """
    if not winners:
        return (
            CHAMPIONS_HEADER
            + "_No FARM-EXP.3-format run records found yet; the "
            "champions table will populate once runner slices "
            "(FARM-EXP.5+) write run records carrying the canonical "
            "axis block + fmax_samples array._\n"
            + manual_appendix
        )

    eval_sets = sorted({k.eval_set for k in winners})
    out: list[str] = [CHAMPIONS_HEADER]
    for eval_set in eval_sets:
        out.append(f"## eval_set: {eval_set}\n\n")
        tiers = sorted(
            {k.tier for k in winners if k.eval_set == eval_set}
        )
        for tier in tiers:
            out.append(f"### tier: {tier}\n\n")
            out.append(
                "| aspect | shortid | fmax mean | 95% CI | run_id "
                "| axis |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
            )
            aspects = sorted(
                key.aspect
                for key in winners
                if key.eval_set == eval_set and key.tier == tier
            )
            for aspect in aspects:
                key = TriplKey(
                    eval_set=eval_set, tier=tier, aspect=aspect
                )
                champ = winners[key]
                shortid = champ.record.shortid or "(unknown)"
                # Annotate placeholder shortids per FARM-EXP.2
                # placeholder-digest tracker.
                if shortid != "(unknown)" and "placeholder" in (
                    champ.record.axis.get("eval_set_manifest_sha")
                    or ""
                ):
                    shortid = f"{shortid} (placeholder)"
                ci = f"[{champ.ci_lo:.4f}, {champ.ci_hi:.4f}]"
                axis_str = _format_axis(champ.record.axis)
                run_id = f"`{champ.record.run_id}`"
                out.append(
                    f"| {aspect} | {shortid} | {champ.mean:.4f} "
                    f"| {ci} | {run_id} | {axis_str} |\n"
                )
            out.append("\n")
    if manual_appendix:
        out.append(manual_appendix)
    return "".join(out)


# ----------------------------------------------------------- symlinks


def _symlink_relative_target(
    run_dir: Path, symlink_path: Path
) -> str:
    """Compute the relative symlink target string."""
    # ``run_dir`` may not exist on disk yet (synthetic fixtures); we
    # only need string-level relpath, not resolve()-walking.
    return os.path.relpath(run_dir, start=symlink_path.parent)


def _infer_run_dir(champion: ChampionRecord) -> Path:
    """Derive the run's artefact directory.

    Strategy: use the source JSON file's parent. This mirrors the
    lab convention where each run writes ``run.json`` next to the
    rest of its artefacts (``model.lgb``, ``cafaeval/``, ...).
    """
    return champion.source_path.parent


def write_symlinks(
    winners: dict[TriplKey, ChampionRecord],
    symlink_root: Path,
    *,
    dry_run: bool = False,
) -> list[tuple[Path, str]]:
    """Create relative symlinks for every champion. Returns the planned set."""
    planned: list[tuple[Path, str]] = []
    for triple, champion in winners.items():
        link_dir = (
            symlink_root
            / triple.eval_set
            / triple.tier
        )
        link_path = link_dir / triple.aspect
        run_dir = _infer_run_dir(champion)
        target = _symlink_relative_target(run_dir, link_path)
        planned.append((link_path, target))
        if dry_run:
            continue
        link_dir.mkdir(parents=True, exist_ok=True)
        # ``link_path`` may already exist as a symlink, file, or
        # directory; unlink/rm appropriately before recreating.
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink()
            else:
                # Directory: rmtree-equivalent for safety.
                import shutil

                shutil.rmtree(link_path)
        link_path.symlink_to(target)
    return planned


# ----------------------------------------------------------- diff/dry-run


def render_dry_run_summary(
    winners: dict[TriplKey, ChampionRecord],
    traces: dict[TriplKey, list[Decision]],
    planned_symlinks: list[tuple[Path, str]],
) -> str:
    """Human-readable dry-run summary."""
    lines: list[str] = []
    lines.append(f"[dry-run] would write {len(winners)} champion(s):")
    for key in sorted(
        winners,
        key=lambda k: (k.eval_set, k.tier, k.aspect),
    ):
        champ = winners[key]
        promotions = sum(
            1 for d in traces[key] if d.verdict == "promoted"
        )
        rejections = sum(
            1 for d in traces[key] if d.verdict == "rejected"
        )
        lines.append(
            f"  - {key.as_str()}: champion={champ.record.run_id} "
            f"(mean={champ.mean:.4f}, ci=["
            f"{champ.ci_lo:.4f}, {champ.ci_hi:.4f}])"
            f"  promotions={promotions} rejections={rejections}"
        )
    lines.append(
        f"[dry-run] would refresh {len(planned_symlinks)} symlink(s) "
        f"(relative paths):"
    )
    for link_path, target in planned_symlinks:
        lines.append(f"  - {link_path} -> {target}")
    if not winners:
        lines.append(
            "  (no usable run records; champions.md will commit as a "
            "scaffold)"
        )
    return "\n".join(lines) + "\n"


def render_explain(
    triple: TriplKey,
    trace: list[Decision] | None,
) -> str:
    """Verbose decision trace for one triple."""
    out: list[str] = [f"== Explain: {triple.as_str()} ==\n"]
    if not trace:
        out.append("(no runs in this triple)\n")
        return "".join(out)
    for i, d in enumerate(trace, start=1):
        out.append(
            f"step {i}: candidate run_id={d.candidate_run_id} "
            f"shortid={d.candidate_shortid}\n"
            f"  mean={d.candidate_mean:.4f} "
            f"ci=[{d.candidate_ci_lo:.4f}, {d.candidate_ci_hi:.4f}]\n"
            f"  verdict={d.verdict}  reason: {d.reason}\n"
        )
    return "".join(out)


# ----------------------------------------------------------- top-level


def run(
    *,
    runs_dir: Path,
    champions_file: Path,
    symlink_root: Path,
    apply: bool,
    strict: bool = False,
) -> tuple[
    dict[TriplKey, ChampionRecord],
    dict[TriplKey, list[Decision]],
    list[tuple[Path, str]],
]:
    """End-to-end pipeline. Returns (winners, traces, symlink_plan)."""
    pairs = _load_records_with_source(runs_dir, strict=strict)
    buckets = group_by_triple(pairs)
    winners, traces = compute_champions(buckets)

    # Preserve the manually-curated appendix from the existing file
    # so pre-FARM-EXP.3 manual entries survive ``--apply`` re-renders.
    manual_appendix = ""
    if champions_file.exists():
        try:
            manual_appendix = extract_manual_appendix(
                champions_file.read_text()
            )
        except OSError:
            manual_appendix = ""

    if apply:
        champions_file.parent.mkdir(parents=True, exist_ok=True)
        champions_file.write_text(
            render_champions_md(winners, manual_appendix=manual_appendix)
        )
        planned = write_symlinks(
            winners, symlink_root, dry_run=False
        )
    else:
        planned = write_symlinks(
            winners, symlink_root, dry_run=True
        )
    return winners, traces, planned


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Champion tracker for FARM-EXP.4.",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="directory holding run-record JSON files (recursive)",
    )
    p.add_argument(
        "--champions-file",
        type=Path,
        default=DEFAULT_CHAMPIONS_FILE,
        help="path to champions.md (default: repo-root/champions.md)",
    )
    p.add_argument(
        "--symlink-root",
        type=Path,
        default=DEFAULT_SYMLINK_ROOT,
        help="root for the runs/champion/<eval_set>/<tier>/<aspect> tree",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "write champions.md and refresh symlinks. Without this "
            "flag the script prints what it would do (dry-run)."
        ),
    )
    p.add_argument(
        "--explain",
        default=None,
        help=(
            "print the decision trace for EVAL_SET:TIER:ASPECT and "
            "exit (does not write anything)."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "error out on malformed run records instead of warning + "
            "skipping."
        ),
    )
    args = p.parse_args(argv)

    if args.explain is not None:
        try:
            triple = TriplKey.parse(args.explain)
        except ValueError as exc:
            print(f"[err] {exc}", file=sys.stderr)
            return 2
        pairs = _load_records_with_source(
            args.runs_dir, strict=args.strict
        )
        buckets = group_by_triple(pairs)
        _, traces = compute_champions(buckets)
        sys.stdout.write(render_explain(triple, traces.get(triple)))
        return 0

    winners, traces, planned = run(
        runs_dir=args.runs_dir,
        champions_file=args.champions_file,
        symlink_root=args.symlink_root,
        apply=args.apply,
        strict=args.strict,
    )
    if args.apply:
        print(
            f"[apply] wrote {args.champions_file} with "
            f"{len(winners)} champion(s); refreshed "
            f"{len(planned)} symlink(s)."
        )
    else:
        sys.stdout.write(
            render_dry_run_summary(winners, traces, planned)
        )
    return 0


__all__ = [
    "ChampionRecord",
    "Decision",
    "TriplKey",
    "compute_champions",
    "group_by_triple",
    "main",
    "render_champions_md",
    "render_dry_run_summary",
    "render_explain",
    "run",
    "write_symlinks",
]


if __name__ == "__main__":
    sys.exit(main())
