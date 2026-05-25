# ADR D39: F-DATA-PACK loop — FAIR dataset packaging for the 24-dataset grid

**Status:** Accepted
**Date:** 2026-05-25

## Context

FARM-EXP.13 is materialising a 24-dataset grid across 8 PLMs and
K in {3, 5, 10} using the `export_research_dataset` pipeline. Each
dataset is a frozen parquet snapshot containing `train.parquet`,
`eval.parquet`, and `manifest.json`.

Before this loop, the datasets existed as opaque filesystem artefacts
without:
- validated manifests (schema version + content-hash verification),
- per-dataset human-readable READMEs,
- per-PLM dataset cards following the HuggingFace DatasetCard format,
- a FAIR-compliance document covering data lineage, split methodology,
  leakage analysis, and a Findable/Accessible/Interoperable/Reusable
  checklist.

The neural-head track (FARM-NEURAL.*) was evaluated as an alternative
allocation of the same GPU budget and was deferred indefinitely (PROTEA
ADR D38). The pivot to F-DATA-PACK makes the 24-dataset grid a
concrete, externally reproducible artefact that directly supports the
reproducibility requirements for thesis chapter 6.

## Decision

1. **Materialise a manifest validator (F-DATA-PACK.1, lab PR #48).**
   `scripts/validate_manifest.py` checks schema version, required
   fields, and content hashes against the declared `schema_sha` and
   `manifest_sha`. It is wired as a CI gate so that any future dataset
   that fails validation is blocked before it enters the training
   pipeline.

2. **Auto-generate per-dataset READMEs (F-DATA-PACK.2, lab PR #49).**
   `scripts/generate_dataset_readme.py` reads `manifest.json` and
   emits a `datasets/<name>/README.md` covering the dataset name,
   PLM, K, eval window, producer version, producer git SHA, schema SHA,
   manifest SHA, and row counts. Eleven cells were emitted on first run.

3. **Ship 8 per-PLM dataset cards (F-DATA-PACK.3, lab PR #50).**
   `dataset_cards/<plm>_card.md` follows the HuggingFace DatasetCard
   YAML front-matter format. One card per PLM in the canonical 8-PLM
   set (see PROTEA ADR D35). Cards are static (not auto-generated) and
   must be updated manually if the embedding configuration changes.

4. **Publish a FAIR/coverage provenance document (F-DATA-PACK.4, lab PR #51).**
   `docs/dataset_provenance.md` covers: data sources (GOA window, 13
   training pairs), pipeline version (producer version + git SHA +
   schema SHA), the canonical 8-PLM embedding config table, split
   methodology (temporal lineage deltas, v226-v230 eval), PCA fit
   policy (transductive, accepted 2026-05-20), leakage-free analysis
   (anc2vec replication artefact fix in PROTEA commit 223299c), and
   a full FAIR checklist per criterion.

5. **Zenodo/HuggingFace Hub deposit (F-DATA-PACK.5, pending).**
   The 24-dataset grid will be deposited on Zenodo after FARM-EXP.13
   completes (13 export jobs in flight as of 2026-05-25). DOI
   assignment is a prerequisite for the F1/A1 FAIR criteria.

## Consequences

**Positive**

- Thesis chapter 6 and Appendix A can cite the datasets by manifest
  content hash and producer git SHA, satisfying the reproducibility
  requirements for publication review.
- Any external consumer can validate a downloaded dataset against the
  published manifest without trusting the file system state.
- The per-PLM dataset cards are machine-readable by HuggingFace tooling
  and will ease a future Hub deposit.

**Negative**

- Per-dataset READMEs and the provenance document require manual
  updates if the bench-v1 family expands beyond the current 24 cells.
  The generator script handles new cells, but the FAIR checklist and
  PLM table in `docs/dataset_provenance.md` are manually maintained.
- F-DATA-PACK.5 (Zenodo) is a hard dependency for the F1/A1 FAIR
  criteria; until it lands the FAIR checklist carries "Pending" entries.

**Neutral**

- The existing `bench-v1-K5-v226-lineage-prostt5` champion row and its
  three-seed metrics are unchanged. This ADR does not alter any
  `RerankerModel` registry row or cafaeval result.
- The F-DATA-PACK loop is independent of FARM-EXP.13; packaging scripts
  work on whatever cells are locally present. The full 24-cell grid
  will be packaged once FARM-EXP.13 completes.

## References

- PROTEA ADR D35: canonical 8-PLM embedding config IDs and orphan
  classification.
- PROTEA ADR D38: neural-head deferral and F-DATA-PACK pivot (the
  authoritative decision record for the choice to ship the curated
  dataset grid over a DL competitor).
- Lab ADR D34: selective rerank policy (LightGBM champion design).
- `docs/dataset_provenance.md`: full FAIR/coverage provenance document.
- Memory key `project_pca_transductive_decision_2026_05_20`: PCA fit
  policy accepted for the FARM-EXP.13 grid.
- Memory key `project_anc2vec_leakage_mechanism`: anc2vec replication
  artefact fix narrative for thesis chapter 6.
- Memory key `project_multi_plm_v226_sweep_plan`: 24-dataset grid scope
  (8 PLMs x K in {3, 5, 10}).
