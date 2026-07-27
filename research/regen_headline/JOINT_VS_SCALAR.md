# The reranker was never grafting. It was fusing summaries. Fusing the vectors wins.

Receipt for the architecture question. Scripts `storage/cooc_experiment/joint_vs_scalar.py`
(cross-fitted) and `joint_vs_scalar_temporal.py` (deployable shape), data in the matching `.json`.

## The premise the campaign had wrong

We had been calling our attempts "grafts", as if each signal were bolted onto the outside of the
model. That is not what happened. **The deployed reranker already fuses every signal TransFew
fuses, jointly, in one model**: `eval.parquet` carries 24 sequence and kNN columns, 12
ontology-structure columns, 6 alignment, 3 text and 2 classifier. LightGBM sees all of them at
once. So "+0.0016 for ProtST" was not a graft failing to take; it was a joint model failing to use
a signal it already had.

**The difference from TransFew is dimensionality, not jointness.** We compress each signal to one
or three scalars before fusing (`protst_vote_fraction`, `anc2vec_neighbor_cos`,
`neighbor_vote_fraction`). TransFew attends over the vectors. Whatever a signal carries beyond its
summary statistic, our reranker has never been shown.

## The result

Cross-attention over four raw vectors against LightGBM over the 72 scalars, on pk-bpo, both arms on
the same rows, both scored on the board's real line, both reading `> 0` the same way so the
positivity guard cuts them in the same place.

| run | A, scalars | B, vectors | B - A | interval |
|---|---|---|---|---|
| **temporal** (train v160..v225, early-stop v225-v227, score v227-v230 blind) | 0.16186 | **0.18218** | **+0.02032** | bootstrap 95% CI **[0.01839, 0.02267]**, excludes zero |
| cross-fitted over evaluation proteins (**not deployable**) | 0.22613 | **0.30140** | **+0.07526** | sd 0.01575, positive **10/10** |

**+0.020 is larger than the three withholding levers combined (+0.016), and it is the first
architectural win of the campaign.** The precondition held in both runs.

Vectors used, all t0 or older, so none can see the window being scored: the protein code from the
champion encoder `d8979601` (2048-d), ProtRek text (1024-d), the two-tower GO sparse codes
(1024-d, fit on **v227**, our t0, with a t0-independent text block by its own docstring), and
anc2vec (200-d, the **2020-10** release).

## What the two runs say together, and it is the important part

The same architecture over the same signals gains **+0.075** when it can train in distribution and
**+0.020** when it trains the way we deploy. The difference is not the model. **It is how much data
the model is allowed to see**, and the temporal arm sees **18.0%** of the pk-bpo training rows
(3,143,496 of 17.5M; 5,769 of 51,096 proteins) for one reason: the vectors it needs are not stored.
They are leftovers from three unrelated experiments that happen to overlap our proteins.

**The gap between +0.020 and +0.075 is the price of not having the signal store.**

That reframes Track B. The store was scoped as the way to regenerate the headline from typed
columns. It is also, and more urgently, the thing standing between us and the only architecture
that has ever beaten the reranker. It is what turns "a classifier or an end-to-end model" from a
constraint into a choice: the classifier can train on everything because its features are already
in the parquet, and the end-to-end model cannot because its signals are not stored anywhere.

## What this does not claim

It is not a board number, for the reason in `WE_DO_NOT_REPRODUCE_THE_BOARD.md`. Both temporal arms
sit well below the deployed 0.21269 precisely because they train on 18% of the rows, so **A here is
not the deployed booster and must never be quoted as it**. And +0.020 does not close PK-BP's 0.0762
gap to TransFew on its own.

**What it does buy is a decision:** whether to embed the other 82% of the training proteins. That
is a re-embedding job, not a frozen-data operation, and this measurement is the evidence for or
against paying for it.

## A run of mine that was void, recorded because the record is the point

The first version compared arm B, a `pos_weight`ed logit, against arm A read as a probability
centred at 0.5. With a 2.47% positive rate LightGBM's probabilities cluster near 0.02, so A
submitted almost nothing while B submitted a sensible fraction: the positivity guard was cutting
the two arms in different places, which is two variables rather than one. It produced A=0.08475
against B=0.30415 on fold 0, a spurious **+0.2194**, and it was the most exciting number of the
night for about ninety seconds. **The precondition band [0.18, 0.30] on arm A caught it.** Both arms
now share the reweighting and A is read as a raw margin. The void artefacts are kept as
`joint_vs_scalar_VOID_v1.{json,log}` so nobody cites them.

---

# Full coverage: the data was never missing, and the volume does not explain the gap

Scripts `storage/cooc_experiment/export_d8979601.py` and `joint_full.py`, data `joint_full.json`.

## The premise that was wrong, and it was mine

The 18% figure was measured off `storage/layer_ablation/query_d8979601.npy`, a 7,401-protein array
left over from the layer-ablation experiment. **The vectors were never missing.** `d8979601` is the
deployed encoder (`learned-code:hard-neg:08234f06`) and the live database already held **527,426**
of its codes. A read-only export pulled **84,370 x 2048** in **109 seconds** and covers **100.0%**
of the 51,096 pk-bpo training proteins. The leftover covered 11.3%.

**Nothing needed embedding. I inherited a premise from an artefact instead of asking the system that
produces the data**, and wrote an hour of re-embedding plan for data that already existed.

## The result

Three tokens (protein / GO code / anc2vec, all at 100%), text dropped at **both** coverages so the
only variable is data volume. Temporal split as deployed: train v160..v225, early-stop on
v225-v227, score v227-v230 blind. Arm A retrained at each coverage.

| coverage | train rows | positives | A LightGBM | B joint | B - A |
|---|---|---|---|---|---|
| 18% | 2,875,539 | 20,864 | 0.16186 | 0.17697 | **+0.01511** |
| **100%** | **16,037,532** | **129,043** | 0.16791 | **0.18844** | **+0.02053** |

Two consistency checks worth keeping: arm A at 18% returns **0.16186**, reproducing the earlier
temporal run to the digit, and dropping the text token costs about **0.005** on the same rows
(B 0.17697 against the four-token 0.18218), which isolates that token's contribution for free.

## What it says, and what my own gate got wrong

The gate asked whether the volume effect is **positive**. It is: **+0.00542**. The gate passes.

**The gate was the wrong question.** It named a sign when the quantity that mattered was a
magnitude: the gap being explained is +0.07526 minus +0.02032, about **0.055**, and six times the
data buys **0.0054**, roughly **a tenth of it**. So the honest verdict is not the one the script
prints. **More data helps in the right direction and does not explain the gap.** The
in-distribution +0.075 comes mostly from training on the evaluation window's own labels, not from
seeing more of the past.

**#20: a gate on the sign is not a gate on the claim.** The claim was "volume explains the gap";
the test asked "is volume positive". Discipline #10 again, one level up.

## The comparison the script does not make, and it is the one that decides

**B at full coverage scores 0.18844. The deployed reranker scores 0.21269 on the same cell and the
same line.** The joint model, with every row, is still **below the system we already run**.

It beats arm A by a gated +0.0205, and arm A is **not** the deployed booster: it is a
binary-objective LightGBM, and the deployed lambdarank is 0.045 better. Isolating the architecture
was the point of the design, and the price of that isolation is that neither arm carries what the
deployed recipe carries: the lambdarank objective, **and the 72 scalar features**. B sees three
vectors and not one feature.

So the standing of the idea is: **the shape is promising and it does not yet buy a better system.**
The obvious and cheap next step is the one this design deliberately excluded, and it is also what
TransFew actually does, which is not to choose between summaries and vectors but to fuse both:
**vectors alongside the scalars, under lambdarank, against the deployed booster.**
