#!/usr/bin/env bash
SC=/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad
GAF=/home/frapercan/Thesis2/storage/gaf_cache/goa_uniprot_all.gaf.225.gz
zcat "$GAF" 2>/dev/null | awk -F'\t' -v setf="$SC/all_targets.txt" '
BEGIN{ while((getline a < setf)>0) want[a]=1 }
/^!/ {next}
($2 in want){ if(!seen[$2]){ seen[$2]=1; print $2"\t"$13 } }
' > "$SC/target_taxa2.tsv"
echo "EXTRACT_DONE matched=$(wc -l < "$SC/target_taxa2.tsv")"
