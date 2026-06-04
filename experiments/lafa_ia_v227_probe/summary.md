# F-LAFA-IA.1b v227 probe: IA-weighted eval (prostt5 K3)

Dataset: bench-v1-K3-v227-lineage-prostt5 (train cutoff v227, eval band v227->v230).
Protocol: prop=fill norm=cafa no_orphans ia=IA-swissprot-exp-v227.txt. Headline wFmax = IA-weighted micro Fmax (f_micro_w); S_min = min semantic distance (s).

| cell | arm | internal Fmax | cafaeval Fmax | wFmax | S_min |
|---|---|---|---|---|---|
| nk-mfo | reranker | 0.7147 | 0.7512 | 0.7536 | 2.0991 |
| nk-mfo | knn | 0.6258 | 0.6228 | 0.6201 | 3.0993 |
| nk-bpo | reranker | 0.5182 | 0.5056 | 0.4997 | 6.3988 |
| nk-bpo | knn | 0.3860 | 0.3616 | 0.3474 | 9.3550 |
| nk-cco | reranker | 0.7181 | 0.6846 | 0.5911 | 2.8832 |
| nk-cco | knn | 0.5857 | 0.5484 | 0.4179 | 4.5629 |
| lk-mfo | reranker | 0.6210 | 0.6337 | 0.6712 | 3.5415 |
| lk-mfo | knn | 0.5962 | 0.5920 | 0.6138 | 3.4273 |
| lk-bpo | reranker | 0.4146 | 0.4052 | 0.3505 | 7.3179 |
| lk-bpo | knn | 0.4329 | 0.4079 | 0.3479 | 8.9405 |
| lk-cco | reranker | 0.6803 | 0.6548 | 0.4434 | 3.7834 |
| lk-cco | knn | 0.6396 | 0.6085 | 0.4464 | 4.1574 |
