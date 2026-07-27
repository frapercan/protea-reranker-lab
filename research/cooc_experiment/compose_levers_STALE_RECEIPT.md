# WARNING: compose_levers.json carries two RETRACTED numbers

`compose_levers.py` / `.json` (16 July) is the technique-lever experiment. Two of its recorded
values have since been retracted and it must not be cited:

- `baseline_lambdarank: 0.1255` is the **rankpct artefact**. A `rankpct()` map we added and
  assumed harmless because it is monotone cost 0.088 f_micro_w. The real raw baseline is 0.2131.
  Under `prop=fill` a monotone map is not harmless: `fill` uses 0.0 as the sentinel for "not
  submitted", so any positive-shifting map changes which cells are zero.
- `oracle_ceiling: 0.6077` contradicts cafaeval's own TP rule (TP = membership in the
  **propagated** ground truth, `parser.py:174`) and bounds nothing. The achievable denominator is
  **0.7764**; capture is 28.9%.
- `delta_vs_lambdarank: 0.0866` and `still_short_of_board_gap_by: -0.0106` are both computed
  against 0.1255 and inherit the artefact.

**What survives, and it is load-bearing for the strategy gate:** dropping the classifier-only rows
(267,523 rows, 43.4% of the pool, carrying **11,915 of 15,248 positives = 78%**) scores 0.2121,
essentially the full-pool anchor. The reranker scores those rows so badly that losing 78% of the
positives costs nothing. The strategy gate (`storage/regen_headline/STRATEGY_GATE.md`) shows the
other half of that fact: scored by their OWN classifier logit instead, those same rows are worth
**+0.021 held-out, unanimous across ten folds**.
