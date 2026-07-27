# Text-aligned PLMs as a BP-wall lever: ProtST and ProTrek

Board-faithful evaluation of text-aligned protein language models as an
orthogonal signal for GO transfer, targeting the biological-process (BP) wall
(LK-BPO + PK-BPO), the two cells where the sealed 0.4063 pipeline is not #1.

## Question

The BP wall is evidence-bound: structure conserves molecular function (MF) but
not biological process, and every homology or co-occurrence lever tried so far
came back RED or modest. Text-aligned PLMs are supervised on function
descriptions, so they encode a signal that is orthogonal to sequence and
structure homology. Do they add on BP, and is the effect real or a single-model
artifact?

## Protocol

kNN GO transfer on the temporal substrate (query 7401 to reference 15000),
cosine top-30, cosine-weighted GO vote, cafaeval `f_micro_w`, 9 cells
(NK/LK/PK x MFO/BPO/CCO). Numbers are low by design (raw kNN, 15k reference, no
reranker); the DELTA is the finding, not the absolute. NK is the leakage-free
anchor (those proteins had no function text at t0). Stratified by category
(the 9 cells), by length, and by neighbour-identity (MMseqs2 buckets:
hard = twilight + no-hit + remote, 3090 queries; easy = high + mod, 4311).

Same-base control: for ProtST we also extract the raw ESM-1b mean-pool from the
identical model, so the difference isolates the text-alignment contribution from
the base PLM.

## Result 1: ProtST dents the BP wall (confirmed)

ProtST (ESM-1b + PubMedBERT, text from Swiss-Prot function descriptions via
ProtDescribe, Apache-2.0). `protein_feature`, 512-d, text-aligned.

| config | mean | nk-BPO | lk-BPO | pk-BPO |
|---|---|---|---|---|
| champion d8979601 | 0.2150 | 0.166 | 0.217 | 0.064 |
| esm1b-raw (same base) | 0.1843 | 0.145 | 0.163 | 0.055 |
| protst-text | **0.2455** | 0.206 | 0.235 | 0.091 |
| combined (protst + d89) | 0.2482 | 0.204 | 0.238 | 0.084 |

- protst-text ALONE beats the champion (0.2455 vs 0.2150).
- The isolated text contribution (protst-text minus esm1b-raw, same ESM-1b base)
  is positive on all 9 cells and largest on BP: nk-BPO +0.062 (leakage-free),
  lk-BPO +0.072 (the wall), pk-BPO +0.037.
- Neighbour-identity: on hard homology the BP text delta survives (nk-BPO +0.053,
  lk-BPO +0.044), so the signal is orthogonal to homology, not a proxy for it.

This is the first lever in the campaign to dent the BP wall in a board-faithful
kNN confirm.

## Result 2: the BP lift is ProtST-specific, not a generic text property

ProTrek (ESM2 + PubMedBERT, trimodal incl. structure, MIT). `get_protein_repr`,
sequence-only, 1024-d, L2-normalised. A second text-aligned model with a
DIFFERENT base, run through the identical protocol.

| config | mean | nk-BPO | lk-BPO | nk-CCO | lk-CCO |
|---|---|---|---|---|---|
| champion d8979601 | 0.2213 | 0.176 | 0.223 | 0.314 | 0.319 |
| protst-text | **0.2541** | 0.215 | 0.246 | 0.383 | 0.381 |
| protrek-text | 0.2154 | 0.182 | 0.207 | 0.362 | 0.387 |

- ProTrek does NOT reproduce the BP lift: nk-BPO +0.006, lk-BPO -0.016 (worse on
  the wall), pk-BPO flat. It is also worse on MF (nk-MFO -0.078, lk-MFO -0.107).
- But ProTrek beats the champion on all three CCO cells (nk +0.048, lk +0.068,
  pk +0.041) and that CCO advantage survives on hard homology (nk-CCO +0.037,
  lk-CCO +0.030).

Interpretation: text-alignment is not monolithic. The text SOURCE and objective
decide the signal shape. ProtST (function descriptions) yields a function/BP
signal; ProTrek (trimodal, structure-inclusive, broader text) yields a
localisation/CCO signal and loses fine MF/BP. The honest claim sharpens to
"a function-description-aligned PLM (ProtST) dents the BP wall", not "text models
do". ProtST's result stands; ProTrek simply does not share the mechanism.

## Result 3: the two text signals are complementary

Equal-weight cosine kNN combine:

| combo | mean | nk-CCO | lk-CCO | lk-BPO |
|---|---|---|---|---|
| pst + d89 | 0.2557 | 0.367 | 0.359 | 0.247 |
| pst + ptk + d89 | **0.2650** | 0.382 | 0.388 | 0.257 |

Adding ProTrek on top of ProtST + champion adds on 8 of 9 cells (largest on CCO,
its strength, but also lk-BPO +0.011). The three representations are not
redundant. Raw-kNN lift over the champion is +0.044 (0.2213 to 0.2650), before
any learned reranker. A learned combine should beat this equal-weight sum.

## Design implication

The meta-reranker (ADR-D43) should carry BOTH text scorers as text-to-GO
EvidenceScorer ports: ProtST for BP/MF, ProTrek for CCO.

## Caveats

1. Raw kNN, 15k reference, no reranker. Absolutes are low; the delta is the point.
2. ProtST and ProTrek are supervised on function vs the self-supervised base, so
   part of the advantage is by design. NK is clean on the query side, but the
   models encode the reference proteins' function well because they trained on
   their text. Fair (it is the value proposition) but state it to a committee.
3. combined approx protst-text on the ProtST run: ProtST largely subsumes the
   champion's BP rather than purely complementing it (replace-vs-augment).
4. Single text-model would be an artifact risk; Result 2 addresses it and, more
   usefully, shows WHICH text signal matters.

## Next

Reranker integration on-platform (POST /jobs): port protst-text (+ protrek-text)
as EvidenceScorers, measure the board 9-cell lift over the sealed 0.4063. Signal-
store the text embeddings as new provenance-versioned signals (DB cache, like PLM
embeddings). Optionally a learned k-WTA head on protst-text (representation-
learning principle on the text rep).

## Receipts

`knn_confirm_text_result.json` (ProtST), `knn_confirm_protrek_result.json`
(ProTrek), `knn_triple_result.json` (combine), `stratify_identity_result.json`
and `stratify_protrek_result.json` (neighbour-identity), plus `preds*/` for
re-cafaeval. Scripts: `extract_protst_both.py`, `extract_protrek.py`,
`knn_confirm_text.py`, `knn_confirm_protrek.py`, `knn_triple.py`,
`stratify_identity.py`, `stratify_protrek.py`.
