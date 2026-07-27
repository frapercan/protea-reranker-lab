# Track B step 2/3: backfill the six LAFA columns, then reclaim the jsonb

Measured on the live DB 2026-07-11 ~04:15 (read-only). This turns the "authorized but
scary" backfill into a sized, disk-safe, ready-to-execute plan.

## Ground truth (verified, not assumed)

- `go_prediction`: **52,235,220 rows**, **101 GB total** (95 GB heap + 6.7 GB toast/idx).
- Root disk: **108 GB free** / 915 GB (88% used). A one-shot full-table UPDATE (rewrites
  all 52M rows, transient +~95 GB) would leave ~13 GB and can crash the live stack. NOT safe.
- Live DB alembic revision: **a8b1c2d3e4f5** = the DOWN-revision of #720's migration
  `f9b2c1a4d7e0`. **The migration is NOT yet applied**; the six columns do not exist yet.
- `go_prediction` already has ~50 typed feature columns (identity_nw ... emb_pca_query_0..15,
  anc2vec_*, tax_*). The six from #720 (`classifier_score`, `classifier_present`,
  `self_prior_score`, `association_total`, `association_cross`, `association_present`) are
  genuinely the remaining jsonb-only features. **#720 targeted the right table.**
- The six keys appear in only a MINORITY of rows: **16,026,336 rows (30.7%)** have at least
  one. (Earlier "only `distance`" readings were a harness lie: `LIMIT 1` on
  `jsonb_object_keys` limits KEYS, not rows.)
- `features` jsonb totals **74 GB** across the table: it is a redundant mirror of the ~50
  typed columns plus the six unpromoted ones. Dropping it is the real disk prize (~74 GB).

## Step 2 backfill (SAFE to run autonomously; batched, key-filtered)

Apply the migration first (instant, additive ADD COLUMN nullable, no rewrite) via a detached
worktree off `origin/develop` (NEVER touch repositories/PROTEA) with alembic pointed at the
live DB: `alembic upgrade f9b2c1a4d7e0` (target the exact rev, not `head`).

Then backfill only the rows that carry the keys, batched by id range with a VACUUM between so
each batch's dead tuples are reused (bounds transient bloat to ~one batch, not the table):

```sql
-- per batch, id in [lo, hi) of width ~500k over min(id)=90,300,716 .. max(id)=144,747,267
UPDATE go_prediction SET
  classifier_score    = (features->>'classifier_score')::float,
  classifier_present  = (features->>'classifier_present')::float,
  self_prior_score    = (features->>'self_prior_score')::float,
  association_total   = (features->>'association_total')::float,
  association_cross   = (features->>'association_cross')::float,
  association_present = (features->>'association_present')::float
WHERE id >= :lo AND id < :hi
  AND features ?| array['classifier_score','classifier_present','self_prior_score',
                        'association_total','association_cross','association_present'];
-- then: VACUUM (ANALYZE) go_prediction;   -- NOT FULL
```

Touched rows ~16M; permanent heap growth ~0.8 GB (6 float8 inline). Transient per batch <~1 GB.
Self-guard: before each batch check `df` and ABORT if free < 20 GB; write a durable progress
log; row-level locks only (does not block live reads).

Verify after: the six columns' non-null counts equal the per-key coverage; spot-check a row's
column values equal its `features->>key`. These are NOT the sealed-0.4063 features (that dataset
is the offline export, not this served table), so no sealed number is at risk.

## Step 3 reclaim the 74 GB (NOT a 4am autonomous op)

`ALTER TABLE go_prediction DROP COLUMN features;` is instant (metadata) but reclaims nothing
until a rewrite. Reclaiming the ~74 GB needs `VACUUM FULL` (ACCESS EXCLUSIVE lock, blocks live
/annotate reads) or `pg_repack` (online, needs the extension). Either is a MAINTENANCE-WINDOW
decision with the author, not an unattended sweep. Do `predictions_jsonb` in the same pass if
it is likewise redundant (check first).

## Why deferred tonight

Live production DB serving a public endpoint, author away, conservative charter, and the
imminent at-scale GPU experiment is the stated priority. Step 2 is safe but a 20-35 min live
mutation better run as one deliberate session; step 3 needs a window. Plan is ready to execute.
