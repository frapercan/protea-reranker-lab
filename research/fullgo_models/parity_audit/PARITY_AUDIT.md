# Native vs Offline-champion parity audit

Read-only source/artifact investigation. No live DB, no code changes. Goal:
find every divergence between the NATIVE on-platform reranker pipeline and the
OFFLINE champion that sealed `f_micro_w 0.391` (NK 0.477 / LK 0.482 / PK 0.215)
on the LAFA 7401 frame, and rank them by likely contribution to the residual
gap (native best `0.3626`, gap `-0.0287`, biggest on LK `-0.046` and NK `-0.025`,
PK `-0.016`).

Evidence base:
- Offline champion: `worktrees/native-boosters-lab/fullgo/{ensemble_seal.py,
  train_classifier_m2.py, seed_average.py, assoc_feature.py, rescorer_seal.py,
  config.yaml, REPRODUCE.md, RESULTS.md}` + `storage/fullgo_models/feature_spec.json`.
- Native: `worktrees/protea-deploy/protea/core/operations/predict_go_terms/
  {_post_knn_pipeline.py, _reranker_scorer.py, _association_loader.py}` +
  `worktrees/protea-deploy/protea/core/classifier_producer.py`.
- Native training + seal: `storage/fullgo_models/{train_native_boosters.py,
  native_boosters_*/summary.json, native_boosters_azucar/seal/*.json,
  ablation_a1.json, lean_ia_p_experiment/RESULTS.md, lever_experiments/RESULTS.md}`.

---

## Headline verdict

The single highest-leverage divergence is **NOT the classifier** (that is at
parity), it is that **the self-prior AND cross-aspect association features are
DEAD (zero-gain) in the native training export** the sealed boosters were fit on.
The offline champion's whole `0.358 -> 0.381 -> 0.391` climb is exactly those two
levers (self-prior +0.023, association +0.009/0.010); in the native parquet they
add literally nothing. That, plus the missing `IA` feature, plus a coarser GBM
hyper-set, plausibly accounts for the bulk of the `-0.0287` gap, concentrated on
LK/PK (association's home) and NK (IA's home).

Hard evidence for the dead features: `storage/fullgo_models/ablation_a1.json`
shows `base`, `base+selfprior`, and `base+sp+assoc` have **bit-identical AUC** in
all three categories (NK 0.8949 / LK 0.9057 / PK 0.8240). Adding self_prior and
association to the native feature set changes the model by 0.0000. In the offline
champion those same two features move the sealed mean by +0.033.

---

## Ranked divergences

### 1. Self-prior + association are ZERO-GAIN in the native training export (TOP)

**Offline champion.** `self_prior` (`sp`, `sp_p`) lifts the sealed mean from 0.358
to 0.381 (`RESULTS.md`); `association` (`assoc`, `assoc_x`, `assoc_p`) lifts it
0.381 -> 0.390/0.391. Self-prior = GOA non-experimental t0 propagated leaf +
ancestors; association = `P(t | known t0 term k)` from training co-occurrence with
a cross-aspect split. Both are real, validated SELECT-internal (`select_cv.py`:
+0.0026 for association, leakage-clean, importance exactly 0 on NK).

**Native.** The feature-builder code (`_post_knn_pipeline.py:apply_self_prior`,
`apply_association`) is faithful and leakage-clean. BUT the boosters the seal
actually used (`native_boosters_azucar`, `native_boosters_v5`) were trained on
`datasets/fullgo-union-SELECT-160-220-227-v5`, and in that parquet the
self_prior/association columns carry no signal:

- `ablation_a1.json` (`auc` block): `base` == `base+selfprior` == `base+sp+assoc`
  to 4 decimals in every category. A feature with real signal cannot leave AUC
  bit-identical; these columns are constant/near-constant in the training data.
- This is the documented "Native association=0 bug" (MEMORY: `term_cooccurrence`
  was only built for v227, not the 13 training t0 sets v160..v220, so every TRAIN
  row's association is 0; the GBM learns to ignore the column, then at predict
  time the live v227 association is non-zero but the booster has no split on it).
  Self-prior shows the same dead signature here, so the same "only built/non-zero
  on some sets" failure likely bites it too in this specific export.

**file:line evidence.**
- Offline lever sizes: `worktrees/native-boosters-lab/fullgo/RESULTS.md` trajectory
  table (`0.358 -> 0.381 -> 0.390/0.391`).
- Native dead features: `storage/fullgo_models/ablation_a1.json` `auc.*` (base vs
  base+selfprior vs base+sp+assoc all equal); driven by the v5 dataset named in
  `native_boosters_azucar/seal/run_NK.json` (`dataset.name =
  "fullgo-union-SELECT-160-220-227-v5"`).
- Native feature code is correct: `_post_knn_pipeline.py:107` (`apply_self_prior`),
  `:209` (`apply_association`), `:363` (`_accumulate_association`, `P(t|k)=count/f`,
  cross-aspect by `aspect_by_go`) -- this MATCHES `assoc_feature.py:8-9`
  (`a_all = sum_k cooc[k,t]/freq[k]`, `a_cross` = cross-aspect only).

**Impact estimate: HIGH, ~0.012-0.020 of the gap.** Association alone is worth
+0.009/+0.010 on the offline TEST frame and is the PK/LK driver; self-prior is
worth far more offline (+0.023 going 0.358->0.381) though much of that may already
be captured natively via KNN/`knn_present` overlap. If both columns are dead in
training, the native boosters are effectively a "KNN + classifier + v6" model with
no prior-knowledge transfer -- which is precisely the LK/PK-shaped hole observed.

**Fix difficulty: MEDIUM.** Re-export the training parquet with co-occurrence (and
the self-prior source) built for ALL 13 t0 training sets v160..v220, not just v227,
then retrain. The `restamp_association*.py` scripts in `storage/fullgo_models/`
were an attempt at this; the durable fix is the export-time build. **This needs a
value-level check**: confirm by reading the v5 train.parquet whether the
`association_total` / `self_prior_score` columns are all-zero on the TRAIN rows
(eval rows being non-zero is the tell). The ablation AUC equality already strongly
implies it.

---

### 2. Missing `IA` feature (and the GBM never sees information accretion)

**Offline champion.** `IA` (information accretion weight per candidate term, from
`IA.tsv`) is one of the 16 features (`feature_spec.json`; `ensemble_seal.py:144`
`ia.get(t, 0.0)`). It is a real lever -- it is the eval weighting itself surfaced
as a feature, so it lets the GBM prefer terms that the metric rewards.

**Native.** Not in the lean/v5/azucar feature lists. The native boosters never see
IA. The offline-validated de-risk (`lean_ia_p_experiment/RESULTS.md`) measured
adding IA to the native lean set: **+0.0065 mean** (NK +0.0099, LK +0.0067, PK
+0.0031), biggest exactly on NK where native trails most.

**file:line evidence.** `feature_spec.json` `"IA"`; `ensemble_seal.py:144`. Absent
from `native_boosters_azucar/summary.json` `features` and
`lean_ia_p_experiment/apply_and_score.py:27` `LEAN`. Measured lift:
`lean_ia_p_experiment/RESULTS.md` headline table.

**Impact estimate: MEDIUM, ~0.006-0.008 of the gap** (already measured offline on
the native frame; trustworthy delta).

**Fix difficulty: LOW.** Add a per-candidate `IA = IA[go_term_id]` lookup to the
native feature builder, retrain. Already greenlit in the experiment's
recommendation. Note: `lfreq` (`log1p(t0-pool freq)`) IS mirrored natively by
`go_term_frequency`, and the offline `*_p` presence flags ARE mirrored by the
native `knn_present` / `classifier_present` / `association_present` -- so IA +
`sp_present` are the only genuinely missing champion features. `sp_present` was
measured dead (drop it).

---

### 3. GBM hyperparameters and training recipe differ

**Offline champion** (`ensemble_seal.py:166-168`, `config.yaml:35-43`):
`objective=binary, lr=0.05, num_leaves=31, min_data_in_leaf=50,
feature_fraction=0.9, num_boost_round=300` (FIXED, no early stopping, no bagging),
on **16 features**, fit on SELECT then sealed on TEST.

**Native** (`train_native_boosters.py:43-49`, `native_boosters_azucar/summary.json`):
`num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
bagging_freq=1, metric=auc, up to 5000 rounds with early_stopping=50 on the eval
split`, on **30 (lean) to 69 (azucar) features**. Azucar refits on TRAIN+EVAL
concatenated at the v5 early-stop `best_iter` (NK 542 / LK 170 / PK 769).

Divergences that matter: (a) `num_leaves=63` + `min_data_in_leaf=100` + many more
features is a much higher-capacity model fit by AUC-early-stopping on the SELECT
eval split, which can overfit the SELECT frame and transfer worse to TEST than the
offline's deliberately small, fixed-round 16-feature GBM; (b) the offline GBM's 16
features are a curated low-variance set, whereas the native 69-feature azucar set
adds v6/emb_pca/interpro/lineage that were independently re-refuted as non-durable
for the champion (MEMORY: SVD label-emb, GCN, IEA, freq-heads all rejected). More
features fit by AUC early-stop is not obviously better and is a plausible source of
TEST-frame optimism / LK regression.

**Impact estimate: LOW-MEDIUM, ~0.003-0.008**, hard to sign. Could be net negative
on LK (extra capacity overfitting SELECT) or net zero. **Needs a value-level
check**: train a native booster with the offline 16-feature-equivalent set and the
offline `num_leaves=31/min_data=50/300-fixed` recipe and compare on the TEST seal.

**Fix difficulty: LOW** (just hyperparameters), but requires a retrain + reseal to
measure.

---

### 4. Eval harness: TOI applied offline, NOT in the native seal

**Offline champion** (`ensemble_seal.py:175-178`, `config.yaml:50`): cafaeval with
`-toi groundtruth_terms_of_interest.txt` for ALL categories AND
`exclude=groundtruth_PK_known.tsv` for PK, Sep_2025 OBO,
`prop=fill, norm=cafa, no_orphans, th_step=0.01`.

**Native seal** (`native_boosters_azucar/seal/eval_payload.json` `_notes`): "NO
toi_file: matches baseline". The default native eval omitted TOI. The payload notes
even flag this: "OPTIONAL stricter variant: add toi_file ... for max LAFA-parity /
closest to offline 0.391 harness."

This is a real harness divergence, but it cuts BOTH ways and is a measurement
artifact rather than a model gap: the brief's native `0.3626` and the offline
`0.391` must be compared under the SAME harness or the gap is partly illusory.
TOI restricts the scored term universe; whether it raises or lowers `f_micro_w`
depends on the frame. The offline validation experiments (`apply_and_score.py:199`,
`lean_ia_p_experiment`) ALSO ran no-TOI, so the relative deltas there are clean,
but the absolute native-vs-champion gap is contaminated by this mismatch.

**Impact estimate: MEASUREMENT-ONLY, unknown sign.** Could explain a chunk of the
nominal gap with zero model change.

**Fix difficulty: TRIVIAL.** Re-score the existing native seal prediction set WITH
`toi_file` set (the eval_payload already documents the exact path) and re-read the
true apples-to-apples gap before attributing anything to the model. **Do this
first** -- it is free and de-confounds everything below it.

---

### 5. Classifier parity -- AT PARITY (the brief's prime suspect is exonerated)

**Offline champion classifier** (`feature_spec.json`: "M2 anc2vec seed-avg, 7 seeds
base/7/137/23/91/31/53, consensus union top-100, score=sum/n").
`train_classifier_m2.py`: `Hybrid` nn.Module = 2-layer trunk
(`Linear->LayerNorm->GELU->Dropout(0.2)` x2, hidden=1024) + independent head
`Linear(hidden,V)` + label head `scale * (proj(h) @ anc2vec_label_matrix.T)`,
forward = `indep(h) + scale*label`, per-term score = sigmoid. Input = 6-PLM concat
(8320-d, order Ankh-base/ESM2-3B/Ankh-large/ESM2-650M/ESMC/ProtT5), standardised
with train mu/sd, ASL loss, 30 epochs. Emit top-100 per protein, `score>=0.01`.
`seed_average.py`: union of (protein,term) across 7 seeds, score = sum / n_seeds.

**Native classifier** (`classifier_producer.py`). Verdict: **faithful port, and
seed-averaging IS engaged in production.**
- `build_hybrid` (`:99-127`) reproduces the exact `Hybrid` module (same trunk,
  same `indep`/`proj`/`scale`, same `indep(h) + scale*(proj(h)@Lt.T)` forward).
- `_label_matrix_for_vocab` (`:130-144`) rebuilds the L2-normalised anc2vec label
  matrix exactly as training (zero for missing) -- same `anc2vec_2020-10.npz`.
- `PLM_CONCAT_ORDER` (`:58-65`) = the same 6 config ids in the same order, 8320-d,
  matching `config.yaml:6-13`.
- `FullVocabClassifier.predict` (`:178-215`) standardises with checkpoint mu/sd,
  top_n=100, min_score=0.01 -- byte-for-byte the lab's `argsort -> top100 ->
  >=0.01` emit (`train_classifier_m2.py:64-65`).
- `SeedAveragedClassifier` (`:241-294`) implements the union + sum/n consensus
  exactly (`seed_average.py:19-29`).
- **The 7 seed checkpoints exist** (`storage/fullgo_models/seeds/seed_{0..6}_*.pt`,
  seeds base/7/137/23/91/31/53) and **`PROTEA_CLASSIFIER_SEED_DIR` is set to that
  dir in `~/.secrets/protea.env:15`**, so `get_classifier()` returns the
  `SeedAveragedClassifier`, which is what `apply_classifier`
  (`_post_knn_pipeline.py:629`) calls. So the native predict path uses the 7-seed
  seed-averaged M2 classifier, NOT a single seed.

The only caveat is a **deployment/env risk, not a code mismatch**: if the predict
worker process does not inherit `PROTEA_CLASSIFIER_SEED_DIR` (e.g. a container or
systemd unit that does not source `~/.secrets/protea.env`), `get_classifier`
silently falls back to the single `classifier_m2_anc2vec.pt` checkpoint and the
classifier becomes 1-seed (offline single-seed = 0.358 vs 7-seed 0.391). **Needs a
runtime check**: confirm the env var is actually present in the predict worker's
environment for the run that produced 0.3626.

**Impact estimate: ~0 IF the env var is set in the worker (most likely), up to
~0.01 if it silently fell back to single-seed.** Verify the env, then close.

**Fix difficulty: N/A (parity) / TRIVIAL (ensure env propagates to the worker).**

---

### 6. Candidate pool + duplicate collapse -- AT PARITY

**Offline** (`ensemble_seal.py:132-146`, `rows_for`): candidates =
`union(knn, clf, self_prior, assoc)` per protein, KNN row collapsed to the
MAX-score row per (protein,term) (`load_knn:76` `if sc > f[k][0]`), classifier
collapsed by MAX (`load_clf:107`). Scored, then cafaeval collapses duplicate
(protein,term) -- the validation harness uses MAX (`apply_and_score.py:176`
`if s > pcd[key]`).

**Native.** `apply_classifier` (`_post_knn_pipeline.py:602-662`) is a strict UNION:
classifier terms merge into KNN candidates, never remove KNN; existing
(protein,term) gets `classifier_score` set and `classifier_present=1`; new ones
appended. Self-prior/association set columns on existing candidates. KNN K=30
(`native_boosters_azucar/seal/predict_payload.json` `limit_per_entry:30`), same
Ankh-base composite scoring config `bae5ece3` as the offline `canon_composite.tsv`
(`REPRODUCE.md:1`: "Ankh-base K30, composite scoring config bae5ece3"). The native
validation scorer collapses duplicate (protein,term) by MAX
(`apply_and_score.py:176`).

Verdict: candidate union, K=30, scoring config, and MAX-collapse all match. One
subtlety: the offline classifier candidates are a clean per-protein top-100 union;
the native adds ancestor-expansion (`expand_to_ancestors`) BEFORE self_prior/
association/classifier, so the native candidate set is the KNN-leaf-ancestor-closure
union classifier-top-100, which is a SUPERSET of the offline candidate set. More
candidates is generally recall-positive and the MAX-collapse + cafaeval threshold
sweep is robust to it, so this is not a likely gap source.

**Impact estimate: ~0.** **Fix difficulty: N/A.**

---

### 7. Association feature definition -- AT PARITY (when the data is non-zero)

`assoc_feature.py` (offline): `a_all = sum_k cooc[k,t]/freq[k]`, `a_cross` =
cross-aspect-only, known terms capped at `FCAP=1000` (drop freq>1000 as
uninformative), emit TOPN=80 per protein. Native `_accumulate_association`
(`_post_knn_pipeline.py:363-399`): `P(t|k)=count/freq[k]` summed over known k,
cross-aspect via `aspect_by_go`. Definitions match. **One difference worth a
value-level check**: the offline `FCAP=1000` known-term cap and `TOPN=80` emission
cap. The native path applies neither an explicit FCAP nor a TOPN cap in
`_accumulate_association`; it scores every candidate against every known term. This
changes which (protein,term) get a non-zero association and the magnitude
distribution slightly. Likely second-order, but if association is ever revived
(item 1) this should be reconciled. `association_total`/`cross`/`present` names map
to `assoc`/`assoc_x`/`assoc_p`.

**Impact estimate: LOW (~0.001-0.003), and moot until item 1 is fixed.**

---

### 8. Self-prior source/propagation -- DEFINITION PARITY, data suspect

Offline `sp`/`sp_p` = GOA non-experimental t0, propagated leaf+ancestors
(`REPRODUCE.md:4`). Native `apply_self_prior` (`_post_knn_pipeline.py:107-154`)
sets `self_prior_score=1.0` for candidates among the query's OWN pre-cutoff
non-experimental terms, leakage-clean (drops experimental + NOT-qualified). The
native scores the candidate set AFTER ancestor expansion, so the leaf+ancestor
propagation is achieved by expanding candidates first -- definitionally equivalent.
The concern is purely item 1 (the column being dead in the v5 training parquet).

**Impact estimate: folded into item 1.**

---

### 9. Final-GBM seed-averaging, length/taxonomy -- negligible

The offline champion does NOT seed-average the final LightGBM (only the classifier);
native does not either. Length/taxonomy features exist on both sides (native has
more granular alignment NW/SW + tax_voters). No divergence of consequence. PK DART
and scale_pos_weight levers (`lever_experiments/RESULTS.md`) are native-only
explorations (`scale_pos_weight` +0.0020 on PK), orthogonal to parity.

---

## Recommendation (highest-leverage path to close -0.0287)

Do these in order; the first two are free/cheap and de-confound everything:

1. **Re-score the existing native seal WITH `toi_file` set** (item 4) to get the
   true apples-to-apples gap. The offline 0.391 was sealed under TOI; the native
   0.3626 was not. Part of the nominal gap may evaporate with zero model change.
   Trivial, do first.

2. **Verify `PROTEA_CLASSIFIER_SEED_DIR` is live in the predict worker** that
   produced 0.3626 (item 5). If the worker did not source `~/.secrets/protea.env`,
   the classifier silently degraded to single-seed (offline single-seed = 0.358),
   which alone is ~0.01-0.03. Cheap, do second.

3. **THE model fix: re-export the training parquet so self_prior + association are
   NON-ZERO on the 13 t0 training sets, then retrain** (item 1). This is the
   biggest model lever -- the entire offline `0.358 -> 0.391` climb is these two
   features, and `ablation_a1.json` proves they are dead (AUC-identical) in the
   parquet the sealed boosters were fit on. Build co-occurrence (and the self-prior
   source) for v160..v220, not just v227, and re-export. Value-level check first:
   confirm the TRAIN-row columns are all-zero in the v5 parquet.

4. **Add the `IA` feature and retrain** (item 2): measured +0.0065 on the native
   frame, biggest on NK. Low effort, known-positive. Drop `sp_present` (dead).

5. **Optionally tighten the GBM toward the offline recipe** (item 3): test
   `num_leaves=31 / min_data=50 / fixed-rounds` on the 16-feature-equivalent set vs
   the 69-feature azucar set on the TEST seal, to check whether the extra native
   capacity is overfitting SELECT and costing LK.

Negligible / do-not-touch: classifier architecture (parity), candidate union +
K=30 + MAX-collapse (parity), final-GBM seed-averaging (neither side does it),
DAG-propagated `_p` features (the champion's `_p` is a presence flag already
mirrored natively -- measured, not a gap).
