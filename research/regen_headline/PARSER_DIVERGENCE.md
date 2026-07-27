# The evaluator has two parsers, they disagree by 0.0933, and the board uses the other one

Receipt for PKW.9. Scripts: `storage/cooc_experiment/board_exact_flags.py`,
`parser_path_killtest.py`, `parser_divergence.py`. Data: the matching `.json` files.

## What the board actually runs

`CAFA_forever/modules/local/evaluation.nf:211-215`, identically at 243, 275, 308, 340, 372:

```
cafaeval <obo> <predictionsDir> <groundtruthTsv> \
  -ia <IA.tsv> -out_dir <...> \
  -toi <groundtruth_terms_of_interest.txt> \
  -prop fill -norm cafa -threads N -no_orphans
```

**It never passes `-max_terms` and never passes `-th_step`.** A grep for either over `modules/`,
`workflows/`, `nextflow.config` and `main.nf` returns nothing, so cafaeval's defaults apply:
`max_terms=None`, `th_step=0.01`. Every lab run in `storage/cooc_experiment/` passed
`max_terms=500, th_step=0.001`. The phrase "the board's exact flags", used in the PKW.8 receipt,
was wrong and is corrected here.

`-toi` turns out to be irrelevant: all 3,855 distinct terms in the PK-BP ground truth are already
inside the 38,650-term list, so it can only remove predicted terms from outside it. Measured
effect: **-0.0002**. It was worth checking and it was worth reporting that it changed nothing.

## The fact

`max_terms` selects a code path. `_pred_parser_legacy` runs only when a cap is set; otherwise
`_pred_parser_vectorised` runs. A cap of `10**9` can never bind on our file, so it is
semantically identical to no cap and isolates the path as the single variable:

| path | `max_terms` | f_micro_w | pr | rc | cov_w | tau |
|---|---|---|---|---|---|---|
| legacy | `10**9` (never binds) | **0.21293** | 0.1804 | 0.2597 | 0.9273 | 0.398 |
| vectorised | `None` | **0.11960** | 0.0829 | 0.2147 | 0.9811 | 0.004 |

**0.0933, and it is purely the parser.** The cap plays no part: only 5 of 4,455 proteins exceed
500 candidates and they carry 0.4% of rows.

## The cause, tested rather than argued

`_pred_parser_legacy` stores a value only when `prob_f > old` with `old = 0.0`
(`parser.py:230`), so our non-positive scores are silently dropped and their cells stay exactly
zero, which is the sentinel `prop=fill` overwrites with the best descendant's score. The
vectorised path applies a group-max and scatters, with no positivity guard, so those cells
become small stored values that `fill` then protects, blocking the inheritance.

Prediction: removing the non-positive rows from the file must make the two paths agree.

| arm | rows | f_micro_w |
|---|---|---|
| A legacy, full file | 616,223 | 0.21293 |
| B vectorised, full file | 616,223 | 0.11960 |
| **C vectorised, positives only** | 117,575 | **0.21293** |
| D legacy, positives only | 117,575 | 0.21293 |

**|C - A| = 0.00000.** The non-positive rows explain the divergence completely. A, C and D are
identical to five decimals.

## What this does and does not change

**PKW.8 stands, unchanged.** Above zero the two paths agree exactly, because the legacy parser
discards non-positives anyway, so filtering changes nothing for it. PKW.8's baseline is
`tau_pre = 0`, precisely where the paths already coincide, and its sweep never uses negative
thresholds. The **+0.0092 (unanimous, 10 of 10 folds) is valid on the board's real path.**

**What is new is a risk, not a gain.** Submitting non-positive scores to the board costs 0.0933,
and throughout this campaign the legacy parser has been silently protecting the lab from that
mistake. Any submission carrying raw lambdarank margins would lose roughly 0.09 with nothing to
announce it. The board's own row for us (`predictions_protea.tsv`, f_micro_w 0.21807, optimal tau
0.06) implies a probability-like scale, so the deployed path evidently transformed the margins.
**We cannot verify what it does today: the file exists nowhere and its producer is unfindable**
(`predictions_uneval` and `predictions_by_window` are both absent from disk; no repo has ever
tracked the name; the only string match in the tree is `book.ts`, the surface that publishes the
number). That is task #42, and this finding gives it teeth.

## The discipline this cost

Discipline #9 says check what the grader actually runs before optimising for a default. Last
night I applied it to exactly one flag: I grepped `-prop fill`, found it, and stopped reading the
command line. Two more flags differed, and one of them was not a flag at all but a switch between
two implementations that disagree by four tenths of the score we are trying to move.

**#11: read the whole command, not the flag you went looking for.**

---

# Scoring our recipe the way the board does: two of three BP cells do not reproduce

Script: `storage/regen_headline/build_submission.py` (+ `build_submission_rescale.py`), data
`build_submission.json` / `build_submission_rescale.json`. It mirrors the board rather than the
lab: one evaluation per knowledge category, against that category's own ground truth, under the
real line (`-toi`, `-prop fill`, `-norm cafa`, `-no_orphans`, **no `-max_terms`, no `-th_step`**).

| cell | guard only | prefilter 0.4 | guard + rescale | prefilter + rescale | board published |
|---|---|---|---|---|---|
| NK-BP | 0.3021 | 0.29733 | 0.29742 | 0.29581 | **0.3374** |
| LK-BP | 0.30376 | 0.30574 | 0.3049 | 0.30741 | **0.4402** |
| PK-BP | 0.21269 | **0.22288** | 0.2126 | **0.22299** | 0.21807 |

**PK-BP reproduces** (-0.0054 against the published figure), and the PKW.8 prefilter lifts it to
**0.22288** on the board's own line, above the published 0.21807. The +0.0102 measured here
matches the +0.0092 held-out mean from the ten-fold cross-fit.

**NK-BP (-0.0353) and LK-BP (-0.1364) do not reproduce.** No arm here comes close to the
published 0.4402 for LK-BP. The most likely reading is that the submitted LK arm carried signals
this parquet does not contain (the ProtST and abstract work), so nothing is necessarily broken,
but **two of our three published BP cells have no script that regenerates them.** That is the
same escalation as `predictions_protea.tsv`, now with a size.

## A hypothesis of mine, tested and refuted

cafaeval sweeps tau over [0,1] and lambdarank margins run to 10.9, so a raw-margin submission
strands part of its own positives permanently above every threshold. Measured share above 1.0:
pk 3.40%, nk 34.32%, lk 61.75%. The deviation from the board tracked that ordering exactly
(-0.005 / -0.035 / -0.136), which is a tidy story, and it is wrong. Rescaling the survivors onto
(0.01, 1.0] moves nothing: NK 0.3021 -> 0.29742, LK 0.30376 -> 0.3049, PK 0.21269 -> 0.2126.

Saturation is real as a fact and was not the cause. Three points will fit almost any story.
**Discipline #4 applies to explanations, not only to numbers: a tidy account is a suspect too.**
The refutation does carry a fact worth keeping: spreading LK's top 61.75% out so a threshold can
separate them buys +0.001, so the ordering inside that band carries no usable signal, which is
consistent with the misordering already seen in the atlas.
