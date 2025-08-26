#!/bin/bash

main="GD462.GeneQuantRPKM.50FN.samplename.resk10.txt"

# Landmark genes
awk 'NR==FNR {ids[$2]; next}
     NR>FNR {
         split($1,a,".");
         if (a[1] in ids) print
     }' map_lm.txt "$main" > landmark_genes.txt

# Target genes
awk 'NR==FNR {ids[$2]; next}
     NR>FNR {
         split($1,a,".");
         if (a[1] in ids) print
     }' map_tg.txt "$main" > target_genes.txt

echo "Done! Landmark: $(wc -l < landmark_genes.txt), Target: $(wc -l < target_genes.txt)"
