# Phase D: Champions selected on 227-validation (f_micro_w)

Selection metric: f_micro_w (IA-weighted micro Fmax) on 226->227 delta split.
IA file: datasets/ia/IA-swissprot-exp-v227.txt.
Training data: bench-v1-K{3,5,10}-v226-lineage-{PLM} (NO re-export).
Champion per (K, tier-aspect) = argmax f_micro_w on 227-val.

## NK cells

| K | aspect | champion PLM | f_micro_w | cafaFmax (v227-val) | coverage % | | old v226 PLM | changed? |
|-|-|-|-|-|-|-|-|
| K3 | mfo | esm2_3b | 0.6890 | 0.6896 | 97 | esmc_600m | YES |
| K3 | bpo | esm2_150m | 0.6217 | 0.6506 | 98 | esm2_150m | no |
| K3 | cco | esm2_3b | 0.6693 | 0.7682 | 96 | esm2_650m | YES |
| K5 | mfo | ankh_large | 0.6756 | 0.6868 | 97 | ankh_base | YES |
| K5 | bpo | esm2_150m | 0.5650 | 0.5837 | 98 | esmc_600m | YES |
| K5 | cco | esm2_650m | 0.6531 | 0.7500 | 96 | esm2_650m | no |
| K10 | mfo | ankh_large | 0.6534 | 0.6732 | 97 | ankh_large | no |
| K10 | bpo | ankh_base | 0.5418 | 0.5763 | 98 | ankh_base | no |
| K10 | cco | esm2_150m | 0.6301 | 0.7314 | 96 | prostt5 | YES |

## LK cells

| K | aspect | champion PLM | f_micro_w | cafaFmax (v227-val) | coverage % | old v226 PLM | changed? |
|-|-|-|-|-|-|-|-|
| K3 | mfo | ankh_base | 0.7171 | 0.7172 | 97 | esm2_650m | YES |
| K3 | bpo | prot_t5 | 0.6663 | 0.6898 | 98 | prot_t5 | no |
| K3 | cco | prot_t5 | 0.7044 | 0.7957 | 98 | prostt5 | YES |
| K5 | mfo | esm2_3b | 0.7056 | 0.7151 | 97 | esm2_3b | no |
| K5 | bpo | esm2_650m | 0.6458 | 0.6818 | 98 | esm2_650m | no |
| K5 | cco | esm2_150m | 0.6753 | 0.7766 | 98 | esm2_650m | YES |
| K10 | mfo | prot_t5 | 0.6924 | 0.6990 | 97 | ankh_base | YES |
| K10 | bpo | prostt5 | 0.6117 | 0.6518 | 98 | prostt5 | no |
| K10 | cco | ankh_large | 0.6523 | 0.7613 | 98 | esm2_650m | YES |

## Notes

- f_micro_w: IA-weighted micro Fmax (cafaeval with ia=IA-swissprot-exp-v227.txt).
  This is the LAFA alignment metric, preferred over plain Fmax for 227-val.
- cafaFmax (v227-val): unweighted cafaeval Fmax on the same 227-val split.
- coverage: fraction of v226->v227 delta proteins present in the prediction set.
  K3 cells with legacy eval sets may show lower coverage.
- old v226 PLM: the PLM that had the best plain cafaeval Fmax on the full
  v226-v230 eval set (the old selection criterion).
- changed: YES if the 227-val champion differs from the old v226 selection.
