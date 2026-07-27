# Audit: why THIS export (split13) fought us — for next session

Author's question (2026-07-15): "no recuerdo tantos problemas con este split, que ha cambiado vs otras
ejecuciones?" Short answer: it was not one thing. THREE differences stacked, none present in the prior
"assumable" ankh-only runs.

## What changed vs prior export runs

| axis | prior "assumable" runs | THIS run (protst-global-train227-test230) |
|------|------------------------|-------------------------------------------|
| feature set | ankh-only (`compute_protst=false`) | **`compute_protst=true` (FIRST time)** - #732/#733/#734 |
| protst reference bank | n/a | **~527k vectors RELOADED from the 28GB table PER SPLIT** (no disk cache; ankh uses one) |
| session continuity | one warm session | **spanned 3 reboots** (author shutdowns during the week) -> cold page cache each time |
| query plan | custom (warm, stable) | **flipped to a GENERIC plan post-reboot -> 6TB temp hash-spill, 10h stall** |
| split | any | split13 = v225-v227 = **largest annotation set** = worst case for all of the above |

## The two concrete faults (both now understood)

1. **Generic-plan spill (the 10h stall).** After a reboot the parameterized `sequence_embedding` load
   flipped to a generic plan that hash-joined the whole 28GB table and spilled ~6TB to temp instead of
   using the perfect index. FIX APPLIED + PERSISTENT: `ALTER DATABASE protea SET
   plan_cache_mode=force_custom_plan`. Detail: `project_export_generic_plan_spill_2026_07_15` +
   PATH_A_PROTST_EXECUTION.md FOURTH.

2. **ProtST bank reloaded per split (the ~4h build).** `_protst_text.py::_load_reference_bank` caches the
   ~527k-vector bank in a PROCESS dict keyed by `(config, t0_annotation_set)`; the t0 set changes every
   split -> cache miss -> full DB reload every split, no disk cache (ankh has one via
   `_load_reference_pool_cached`). On cold cache = slow. Detail: PATH_A_PROTST_EXECUTION.md FIFTH.

## Plan for next session (tackle the problem)

1. **Decide the cheaper path first.** The offline A/B ALREADY produced decision-grade protst deltas by
   ENRICHING the sealed platform pool with `apply_protst_text` (no full re-export). Question to settle:
   does the authoritative `run_cafa_evaluation` really need a full `compute_protst` re-export, or can we
   enrich the existing sealed platform dataset the same way? If the latter, we skip the expensive export
   entirely.
2. **If we do re-export:** apply the FIFTH fix (disk-cache the protst bank in `_load_reference_bank`,
   keyed by config_id; re-derive per-t0 GO terms only) -> removes the per-split reload. Keep
   force_custom_plan. Set `PROTEA_EXPORT_MINIJOBS=1` + scale batch workers for parallelism. Run in ONE
   warm session (avoid mid-run reboots).
3. **Sanity first:** a small-N export (1-2 train pairs) to time split-build WITH the fix before committing
   to the full 14-split + eval run.

State at cut: splits 0-12 done on disk; split13 was mid-build (lost on power-off, survives on suspend).
Resume steps in PATH_A_PROTST_EXECUTION.md SHUTDOWN section.
