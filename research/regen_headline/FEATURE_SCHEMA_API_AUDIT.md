# Data structure + API audit: the feature schema, its renderers, and the dead families

Author-requested audit (2026-07-16). Two independent read-only passes, each claim
verified against `origin/develop` rather than the local checkouts, which are stale
(PROTEA local was **63 commits behind**; `feature_family_provenance` does not exist
there at all).

Headline: **the "one source, three renderers" invariant HOLDS.** It is not where the
problem is. The problems are (1) a contracts `main`/`develop` fork that a prior
receipt recorded as reconciled and is not, (2) a governance mechanism from PR #710
that was built general and wired to only four of twenty families, and (3) a
published claim of ours about `emb_pca_query_*` that is wrong.

## 1. The invariant holds, and it holds by construction

All three renderers **derive** from the installed contract; none transcribes it:

| renderer | mechanism |
|---|---|
| Sphinx `feature-docs-table` | `docs/source/_ext/feature_docs_table.py:34` imports `FEATURE_DOCS` / `ALL_FEATURES` / `FEATURE_FAMILIES` at build time |
| `GET /features/registry` | `protea/api/routers/features.py:132` iterates `FEATURE_DOCS.values()` |
| the export columns | `parquet_export.py:53` reads `REGISTRY`, itself built `for name in ALL_FEATURES` (`features/_bindings.py:390`), raising `KeyError` on any unbound column |

Evaluated, not eyeballed, on both refs:

```
origin/develop: FEATURE_DOCS 75 | ALL_FEATURES 75 | missing [] | orphans [] | family disagreements []
origin/main:    FEATURE_DOCS 78 | ALL_FEATURES 78 | missing [] | orphans [] | family disagreements []
```

Within a single ref the drift is **zero**. The source is `feature_schema.py` for names
and families, `feature_docs.py` for prose and status: two modules, kept in agreement by
a lint rather than by construction.

## 2. The real drift: the contracts fork is NOT reconciled

`project_contracts_main_develop_schema_divergence_2026_07_10` records this as **DONE**.
It is not, and the memory has been corrected.

| | `SCHEMA_VERSION` | cols | families | schema sha | protst_text |
|---|---|---|---|---|---|
| contracts `origin/main` (v1.5.0) | `v6` | **78** | 21 | `bcb5453c5a0e` | **yes** |
| contracts `origin/develop` (v1.4.0) | `v5` | **75** | 20 | `4e2c515273d3` | no |

29 commits on `develop` are absent from `main`; 10 on `main` are absent from `develop`.
The 2026-07-10 reconciliation did happen; then `21f300f release: v1.5.0, add protst_text
feature family (#43)` landed on `main` **only**, re-forking it. `origin/chore/reconcile-main-into-develop`
exists and is unmerged.

Why it is not currently breaking anything: **PROTEA pins `main`**
(`pyproject.toml:41` `...@main`; `poetry.lock:3979` `version = "1.5.0"`,
`resolved_reference = 21f300f4...`), and `tests/test_feature_contract.py:126` pins the
**main** digest `bcb5453c5a0e`. So the served API and the exports are consistent at 78.

Why it is a trap anyway: `origin/HEAD -> origin/develop`, so **a fresh clone of contracts
gets v5/75**. Anything built from a `develop` checkout (docs, a lab, a new agent's
worktree) renders 75 features while PROTEA serves 78, and both are internally green.
This is what makes the fork survivable and therefore durable.

**No lint compares the two branches.** Each is self-consistent and its own CI passes:
`test_feature_schema.py` asserts `v6` on main and `v5` on develop. Green on both sides of
a fork is exactly how a fork persists.

## 3. PR #710 built a general mechanism and enrolled four families

The ADR-D45 degeneracy check is real and would have caught both dead families:
`_assert_no_degenerate_families` (`parquet_export.py:495`) raises when a family recorded
`produced` is constant across a split, folding all-NaN into "constant" (`:412`, `:492`).

It never examines them, because it iterates `produced_family_columns`, derived from
`ctx.feature_family_provenance`, and there is exactly **one** construction site in the
repo (`training_dump/_export_features.py:252`, in `build_lafa_family_provenance`):

```python
wiring = (
    ("classifier",  flags.classifier,  _CLASSIFIER_PRODUCER),
    ("self_prior",  flags.self_prior,  _SELF_PRIOR_PRODUCER),
    ("association", flags.association, _ASSOCIATION_PRODUCER),
    ("protst_text", flags.protst_text, _PROTST_PRODUCER),
)
```

Four families, hand-maintained, of the 20 to 21 in `FEATURE_FAMILIES`. `interpro` and
`emb_pca` are not among them, so their columns never enter the checked set. The default
is worse than the omission: `feature_family_provenance` defaults to `()`
(`parquet_export.py:180`), documented as "no family is degeneracy-checked and no
provenance is written". Under the default the entire check is inert.

**The gap in one line: the check is opt-in per family, and the families most likely to be
silently dead are precisely the ones nobody remembered to enroll.** It protects new
families and leaves the forgotten ones unguarded.

## 4. The two dead families, for two different reasons

### `interpro_*` (11 cols): CONFIRMED config gap

`PROTEA_INTERPRO_GO_PRED_PATH` is **commented out** in `worktrees/protea-deploy/.env.local:13`.
The producer reads a TSV named by that env var (`_interpro_features.py:38`), and the
unset path is a deliberate graceful no-op (`:85-91`): set-but-missing raises
`FileNotFoundError`; **unset returns `{}`**, the join is a no-op, and all 11 columns keep
`_interpro_default_fields()` zeros.

The sentinel that was supposed to catch this **does not work**. The docstring at
`_leaf_record_builder.py:326` claims the defaults leave `interpro_present` False "so a
true zero is distinguishable from an absent source". It is not: `interpro_present=False`
is emitted identically whether InterPro was consulted and found nothing, or was never
consulted at all. The one field designed to separate "no evidence" from "no source"
collapses both.

### `emb_pca_query_*` (16 cols): NOT what our receipt says

`storage/feature_necessity/WRITEUP.md` claimed these were "**Populated**, and worth
nothing to the booster" and that retiring them "is defensible on this evidence".
**Both halves are false, and the receipt has been corrected.**

The columns are gated by `use_embedding_pca`, which defaults `False` at **every**
declaration site (`export_research_dataset.py:73`, `_export_features_batch.py:64`,
`_export_knn_batch.py:57`, `export_coordinator.py:74`, `training_dump/_payload.py:113`,
`api/routers/datasets.py:159`). With the flag off, `pca_state` is None and
`_KnnTransferRunner._compute_pca_proj` (`:638-650`) emits NaN **by design**, as its own
docstring says. LightGBM reports zero gain for an all-NaN column because it **cannot
split on it**, not because the signal is redundant.

So the family is **unmeasured, not refuted**. There is no evidence against it, only an
absence of evidence. `gain_report.json` carries no populated-ness statistic and could
never have distinguished the two cases.

There is also a design conflation worth naming: `use_embedding_pca` is documented as a
*retrieval* knob (PCA before KNN: smaller index, slight Fmax loss) yet the same boolean
switches these 16 feature columns. Producing the features currently costs retrieval
quality, which is a good reason nobody ever turned it on.

## 5. Also found

- **Two families render nowhere.** `knn_distance` and `knn_vote` have no `FeatureDoc`
  filed under them (their columns sit under the umbrella `knn` family), so the Sphinx
  directive skips them (`feature_docs_table.py:50`) and the router omits them. The lint
  checks doc -> family but never family -> doc.
- **The drift lint is copy-pasted, and admits it.** `PROTEA/scripts/check_feature_docs.py`
  and `protea-contracts/scripts/check_feature_docs.py` are two hand-mirrored copies
  (docstring lines 13-16), because contracts does not export the logic. The two linters
  can themselves diverge.
- **Nothing renders and compares.** No test builds the Sphinx directive or hits
  `/features/registry` and diffs the outputs. The invariant is upheld by code discipline,
  not by an assertion. A hand-transcribed literal reintroduced into the router would pass
  CI.
- **Stale prose:** `api/routers/features.py:4` says "75 entries as of contracts v1.4.0";
  it serves 78 from v1.5.0. Cosmetic, the code is dynamic.
- **The sha pins names, nothing else.** `compute_schema_sha` = first 12 hex of SHA-256
  over `"|".join(sorted(columns))`. No types, no producers, no order.
  `feature_docs.py:35` states the consequence plainly: a booster passes the guard "even
  though the columns carry no measurement". That is the D45 seam, restated.

## 6. Proposals (additive only; NOT applied)

Ranked by what they would have caught.

1. **A completeness test over family provenance.** Assert every key of
   `FEATURE_FAMILIES` appears in the export's provenance tuple in exactly one state.
   Converts "forgot to enroll a family" from a silent all-zero column into a test
   failure. **Would have caught both dead families.** Additive: one test, plus the
   provenance rows it forces.
2. **Enroll `interpro` and `emb_pca` now**, extending `build_lafa_family_provenance`:
   `PRODUCED` when the producer is live, `DECLARED_ABSENT` otherwise. This is the
   mechanism's documented intent (`parquet_export.py:523`: "If the producer is
   intentionally unwired, record the family as declared-absent instead of shipping a
   constant column"). Makes the config gap self-announcing in the manifest.
3. **A cross-branch contracts guard.** A CI job asserting `main` and `develop` agree on
   `ALL_FEATURES` (or that a documented, dated exception exists). Nothing today compares
   them; each is green alone.
4. **Fix `interpro_*` by config**, not code: materialise the TSV, uncomment
   `PROTEA_INTERPRO_GO_PRED_PATH`, re-export. Pure config.
5. **Split `use_embedding_pca`** into the retrieval knob and a new
   `emit_emb_pca_features` (default False), so the 16 columns can be produced without
   paying the retrieval cost. **This must precede any retirement decision on the family**,
   since per §4 it has never been evaluated.
6. **A family -> doc lint** (catches `knn_distance` / `knn_vote`), and a render-and-compare
   test for the three renderers.
7. **Deprecation vocabulary in contracts.** There is none: grep for `deprecat` over
   `feature_schema.py` returns nothing. The schema can say "declared" and nothing else,
   so it cannot express "declared, but no producer is wired in this deployment". An
   optional `DORMANT_FEATURES` set would be additive. **Hold off on marking either family
   deprecated:** `interpro_*` is one config line from live, and `emb_pca_query_*` has no
   evidence against it.

## What this audit did not establish

- The exported parquet was not read directly; "all-NaN in the latest export" is taken
  from the earlier measurement and corroborated by the flag defaults, which agree.
- Whether any historical export ever passed `use_embedding_pca=True` is not determinable
  from code. The flag is recorded per-export (`export_research_dataset.py:352`), so the
  run manifests would settle it.
- Which contracts branch is *intended* to be canonical is an author call. `origin/HEAD`
  says `develop`; PROTEA's pin and its golden test say `main`.
