# Reverse index demo: GO-BP term -> held-out eval proteins

Model: bipartite kWTA link-prediction (checkpoint `bipartite_kwta_ckpt.pt`, best val_mrr 0.4165), NO retraining. Score = dot(kwta(P(code)), kwta(T[term])).

Universe = 5,236 eval proteins held OUT of training. A check (Y) means the eval protein is truly annotated with the term (propagated v230 gt).

Terms shown span IA (rarity) buckets. `sup` = training edges the term had; `n_true` = truly-annotated eval proteins; `P@10`/`AUC` are this term's retrieval quality.


## GO:0009987 — cellular process

IA 0.41 | train-support 43,943 | n_true_eval 3414 | P@10 1.00 | AUC 0.636

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P32728 | Y | 101.958 |
| 2 | I1S104 | Y | 99.982 |
| 3 | P04152 | Y | 91.598 |
| 4 | A0A4V8H042 | Y | 80.181 |
| 5 | O55142 | Y | 78.351 |
| 6 | D7A0Y0 | Y | 74.296 |
| 7 | F9VNG5 | Y | 74.125 |
| 8 | Q47155 | Y | 73.365 |
| 9 | O43036 | Y | 72.869 |
| 10 | Q9H3J6 | Y | 72.326 |

## GO:1904951 — positive regulation of establishment of protein localization

IA 0.45 | train-support 354 | n_true_eval 13 | P@10 0.00 | AUC 0.396

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P43557 | . | 29.321 |
| 2 | O13516 | . | 27.994 |
| 3 | Q1XG90 | . | 27.753 |
| 4 | P62911 | . | 26.706 |
| 5 | A0A0H2ZFV1 | . | 26.565 |
| 6 | Q9C0Y2 | . | 25.824 |
| 7 | Q9D823 | . | 25.418 |
| 8 | I1S104 | . | 25.010 |
| 9 | A0A1D6F7U7 | . | 24.703 |
| 10 | Q59WF4 | . | 24.653 |

## GO:2000143 — negative regulation of DNA-templated transcription initiation

IA 1.24 | train-support 27 | n_true_eval 5 | P@10 0.00 | AUC 0.670

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P04152 | . | 28.587 |
| 2 | P32728 | . | 28.232 |
| 3 | P65870 | . | 27.734 |
| 4 | Q03456 | . | 27.156 |
| 5 | Q47155 | . | 25.177 |
| 6 | O13516 | . | 23.584 |
| 7 | P62918 | . | 23.354 |
| 8 | P58965 | . | 23.109 |
| 9 | P23256 | . | 22.697 |
| 10 | P49078 | . | 22.611 |

## GO:0043412 — macromolecule modification

IA 2.2 | train-support 2,643 | n_true_eval 846 | P@10 0.70 | AUC 0.653

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | Q09709 | Y | 52.948 |
| 2 | A2ADA5 | . | 51.445 |
| 3 | Q8N0Z8 | Y | 50.237 |
| 4 | Q9UJJ7 | Y | 49.298 |
| 5 | P32728 | . | 47.990 |
| 6 | O94295 | Y | 47.508 |
| 7 | Q4QR99 | Y | 46.910 |
| 8 | Q9UUC7 | . | 46.349 |
| 9 | P0A9N4 | Y | 44.860 |
| 10 | Q9UTB3 | Y | 43.597 |

## GO:0061025 — membrane fusion

IA 2.27 | train-support 269 | n_true_eval 15 | P@10 0.10 | AUC 0.794

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P77733 | . | 34.382 |
| 2 | P60508 | . | 29.892 |
| 3 | Q9M2J9 | . | 28.659 |
| 4 | Q1XG90 | . | 27.758 |
| 5 | P0AB38 | . | 27.616 |
| 6 | P53835 | . | 27.600 |
| 7 | Q9BV40 | Y | 27.045 |
| 8 | P0A6N4 | . | 26.908 |
| 9 | Q9CXV1 | . | 26.613 |
| 10 | O23053 | . | 26.239 |

## GO:1990868 — response to chemokine

IA 3.98 | train-support 61 | n_true_eval 5 | P@10 0.00 | AUC 0.787

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P62918 | . | 26.493 |
| 2 | P39176 | . | 26.046 |
| 3 | P0AAX8 | . | 23.485 |
| 4 | P37557 | . | 23.268 |
| 5 | P77733 | . | 22.175 |
| 6 | Q8N0Z8 | . | 21.672 |
| 7 | Q9CXV1 | . | 21.403 |
| 8 | P22287 | . | 21.116 |
| 9 | P14115 | . | 21.114 |
| 10 | Q9D823 | . | 21.105 |

## GO:0065009 — regulation of molecular function

IA 4.58 | train-support 1,034 | n_true_eval 501 | P@10 0.00 | AUC 0.492

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | Q8CEI1 | . | 32.249 |
| 2 | P59936 | . | 30.274 |
| 3 | Q9CXV1 | . | 29.709 |
| 4 | O76454 | . | 29.608 |
| 5 | Q14197 | . | 28.880 |
| 6 | A0A4V8H042 | . | 28.654 |
| 7 | P9WNT5 | . | 28.627 |
| 8 | P59925 | . | 28.573 |
| 9 | P62918 | . | 28.167 |
| 10 | P41057 | . | 27.905 |

## GO:0032989 — cellular anatomical entity morphogenesis

IA 4.8 | train-support 445 | n_true_eval 12 | P@10 0.00 | AUC 0.667

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P62918 | . | 34.470 |
| 2 | Q0CJ62 | . | 31.361 |
| 3 | Q9D1R9 | . | 30.849 |
| 4 | Q9CR57 | . | 30.825 |
| 5 | L0DSL2 | . | 30.783 |
| 6 | Q9Y7Y7 | . | 27.486 |
| 7 | P42628 | . | 27.321 |
| 8 | Q06135 | . | 27.148 |
| 9 | Q1PX48 | . | 26.325 |
| 10 | P61514 | . | 25.743 |

## GO:1990051 — activation of protein kinase C activity

IA 4.36 | train-support 1 | n_true_eval 5 | P@10 0.00 | AUC 0.780

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P50596 | . | 8.664 |
| 2 | Q9CPR4 | . | 7.977 |
| 3 | Q6EBC2 | . | 7.824 |
| 4 | Q9NPF7 | . | 7.677 |
| 5 | Q9Y5G2 | . | 7.095 |
| 6 | P20607 | . | 7.079 |
| 7 | Q8BQ47 | . | 6.647 |
| 8 | A0A1B0GTL2 | . | 6.526 |
| 9 | Q9SF85 | . | 6.492 |
| 10 | Q9URZ7 | . | 6.121 |

## GO:0140462 — pericentric heterochromatin organization

IA 6.79 | train-support 1 | n_true_eval 53 | P@10 0.00 | AUC 0.667

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | Q8RVQ9 | . | 13.897 |
| 2 | O14211 | . | 12.821 |
| 3 | O55142 | . | 11.494 |
| 4 | Q9V6Q2 | . | 11.106 |
| 5 | Q9GZS1 | . | 10.808 |
| 6 | O82533 | . | 10.808 |
| 7 | F4I9G2 | . | 10.522 |
| 8 | Q9Y812 | . | 10.098 |
| 9 | P40215 | . | 10.039 |
| 10 | O34693 | . | 9.863 |

## GO:0050746 — regulation of lipoprotein metabolic process

IA 7.15 | train-support 13 | n_true_eval 8 | P@10 0.00 | AUC 0.408

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | Q9UVF9 | . | 26.225 |
| 2 | P9WPB5 | . | 24.695 |
| 3 | Q79FX8 | . | 24.596 |
| 4 | P9WQD7 | . | 23.397 |
| 5 | Q7U1K1 | . | 23.296 |
| 6 | P02919 | . | 23.152 |
| 7 | P47164 | . | 22.730 |
| 8 | P0AAX8 | . | 22.408 |
| 9 | P11960 | . | 21.767 |
| 10 | Q05778 | . | 21.003 |

## GO:0090069 — regulation of ribosome biogenesis

IA 6.08 | train-support 23 | n_true_eval 5 | P@10 0.00 | AUC 0.429

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | Q9UJJ7 | . | 29.458 |
| 2 | P61514 | . | 28.390 |
| 3 | O55142 | . | 27.625 |
| 4 | Q9CZM2 | . | 25.310 |
| 5 | P9WMX3 | . | 25.187 |
| 6 | P04786 | . | 24.887 |
| 7 | Q9D1R9 | . | 24.072 |
| 8 | P61255 | . | 24.065 |
| 9 | O94242 | . | 22.504 |
| 10 | Q09709 | . | 22.456 |

## GO:0043473 — pigmentation

IA 8.11 | train-support 189 | n_true_eval 17 | P@10 0.00 | AUC 0.703

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | Q9CR57 | . | 32.385 |
| 2 | P0ABN5 | . | 31.608 |
| 3 | P76472 | . | 28.451 |
| 4 | P0A6J1 | . | 27.866 |
| 5 | P65870 | . | 27.861 |
| 6 | G4VQX9 | . | 27.729 |
| 7 | Q10341 | . | 26.868 |
| 8 | A2ADA5 | . | 26.513 |
| 9 | Q84TV3 | . | 26.437 |
| 10 | P53095 | . | 26.193 |

## GO:0140253 — cell-cell fusion

IA 9.21 | train-support 79 | n_true_eval 9 | P@10 0.00 | AUC 0.746

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P60508 | . | 39.348 |
| 2 | A8YXZ8 | . | 28.037 |
| 3 | Q5R676 | . | 28.037 |
| 4 | Q8N4H5 | . | 28.037 |
| 5 | Q9UQF0 | . | 25.793 |
| 6 | B1AXP6 | . | 24.606 |
| 7 | P53697 | . | 24.291 |
| 8 | P0AAJ8 | . | 23.537 |
| 9 | P28241 | . | 21.248 |
| 10 | O55142 | . | 20.875 |

## GO:0043335 — protein unfolding

IA 10.95 | train-support 22 | n_true_eval 5 | P@10 0.00 | AUC 0.578

| rank | eval protein | truly annotated | model score |
|---:|:---|:---:|---:|
| 1 | P40185 | . | 27.869 |
| 2 | P0AFL3 | . | 26.037 |
| 3 | P0A908 | . | 24.844 |
| 4 | P77733 | . | 22.683 |
| 5 | P62918 | . | 22.346 |
| 6 | P61255 | . | 22.185 |
| 7 | P0A9N4 | . | 21.934 |
| 8 | Q9USM3 | . | 21.852 |
| 9 | D7A0Y0 | . | 21.397 |
| 10 | P40711 | . | 21.306 |
