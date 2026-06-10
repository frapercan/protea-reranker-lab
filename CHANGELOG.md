# Changelog

All notable changes to protea-reranker-lab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The repository is currently untagged; the first tagged release will
formalise the version numbers below as part of a coordinated stack release.

## [Unreleased]

### Added

- Substantial Sphinx documentation upgrade: a stage-by-stage workflow
  page (pooled staging, features, LambdaMART training, calibration,
  evaluation), a quickstart, and a contributing guide.
- Full API reference: every public module is now autodoc-ed and grouped
  by pipeline stage (previously 13 of 27 modules were documented).
- `myst-parser` and `sphinx-copybutton` wired into the docs build.
- `CHANGELOG.md` (this file).

### Changed

- Documentation now builds clean under `sphinx-build -W` (warnings as
  errors); `napoleon_use_ivar` removes duplicate dataclass attribute
  descriptions and five module docstrings were corrected to valid
  reStructuredText.
- `mypy` configuration added to `pyproject.toml`; the package now passes
  `mypy` with zero errors. Native dependencies (`pyarrow`, `lightgbm`,
  `wandb`) are treated as untyped so their import-time namespaces no
  longer raise spurious `attr-defined` errors.

### Fixed

- Genuine type-annotation issues surfaced by the new `mypy` gate:
  loop-variable reuse in `propagation`, an `Any`-typed calibrator
  estimator, a `Literal` `val_strategy` on the universal run spec, a
  `Sequence` manifest-path parameter, and widened evaluation dict value
  types.

### Internal

- `_SrcEncoder` construction in `pooled_staging` deduplicated: the
  layout-based constructor now delegates to the cat-codes constructor.
  Behaviour-preserving; the int-coded pooled staging and the four-tuple
  LambdaRank group key are unchanged.

## [0.3.0]

- Universal booster pipeline (F-RERANK-UNIVERSAL): pooled multi-manifest
  staging with int-coded string columns and a global row budget, the
  four-tuple LambdaRank group key, IA-aligned training, per-aspect
  calibration, and the held-out-band `f_micro_w` evaluator. See
  [ADR D41](docs/adr/D41-universal-reranker.md).
- FAIR dataset packaging (F-DATA-PACK): manifest validator, per-dataset
  README generator, per-PLM dataset cards, and the provenance document.

## [0.2.0]

- Per-cell champion pipeline: streaming staging, LightGBM training and
  inference over parquet buckets, numpy Fmax evaluation, and the paired
  bootstrap comparison against the KNN baseline. Binary-objective
  multi-seed champion published to Chapter 6 of the doctoral thesis.

[Unreleased]: https://github.com/frapercan/protea-reranker-lab/compare/develop...HEAD
