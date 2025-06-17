## Data source
## PRJNA1084244
## https://www.ebi.ac.uk/ena/browser/text-search?query=PRJNA1084244%20

## Code source
## https://zenodo.org/records/14030067
## https://zenodo.org/records/14584910

## data_download.sh
## nohup ./data_download.sh > data_download.log &   # 1081321
for srr_id in `cat SRR.txt`;do
    ascp -QT -l 300m -P 33001 -k 1 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh \
         era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/${srr_id:0:6}/0${srr_id:0-2}/$srr_id/$srr_id\_1.fastq.gz .
    echo $srr_id done
done

for srr_id in `cat SRR_raw.txt`;do
    if [ ! -f $srr_id"_1.fastq.gz" ];then
        echo $srr_id >> SRR_2.txt
    fi
done

# SRR28241176 (data not available)



## install minimap2 (version: 2.29-r1283)
conda install bioconda::minimap2

## install bedops
conda install bioconda::bedops

minimap2 -x map-ont -d GRCh38_ONT.mmi /fs/home/jiluzhang/LRS/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz  # ~1.5 min
# -x STR       preset (always applied before other options; see minimap2.1 for details) []
#                  - lr:hq - accurate long reads (error rate <1%) against a reference genome
#                  - splice/splice:hq - spliced alignment for long reads/accurate long reads
#                  - splice:sr - spliced alignment for short RNA-seq reads
#                  - asm5/asm10/asm20 - asm-to-ref mapping, for ~0.1/1/5% sequence divergence
#                  - sr - short reads against a reference
#                  - map-pb/map-hifi/map-ont/map-iclr - CLR/HiFi/Nanopore/ICLR vs reference mapping
#                  - ava-pb/ava-ont - PacBio CLR/Nanopore read overlap
# -d FILE      dump index to FILE []

minimap2 --MD -a -x map-ont -t 8 GRCh38_ONT.mmi SRR28246669_1.fastq.gz > SRR28246669.sam
# --MD         output the MD tag
# -a           output in the SAM format (PAF by default)
# -t INT       number of threads [3]

samtools view -bS -q 30 -@ 8 SRR28246669.sam > SRR28246669_Q30.bam 
samtools sort -@ 8 SRR28246669_Q30.bam -o SRR28246669_Q30_sorted.bam
samtools index -@ 8 SRR28246669_Q30_sorted.bam
samtools rmdup -s SRR28246669_Q30_sorted.bam SRR28246669_Q30_sorted_rmdup.bam
samtools index -@ 8 SRR28246669_Q30_sorted_rmdup.bam

# cat SRR28246669_Q30_sorted_rmdup.bam | bamToBed -cigar | awk -vOFS='\t' \
#                                                          '{match($7, /(^[0-9]+)[SH]/, x)
#                                                            lc=x[1]
#                                                            match($7, /([0-9]+)[SH]$/, x) 
#                                                            rc=x[1]
#                                                            if (lc == "") lc=0
#                                                            if (rc == "") rc=0
#                                                            if (lc < fc && rc < fc) print $1,$2,$3,$4,$6}' \
#                                                            fc=150 |\
#                                        sort -k1,1 -k2,2n | gzip > SRR28246669.bed.gz
                                       
# zcat SRR28246669.bed.gz | awk -vOFS='\t' '{print "chr"$1,$2,$3,"SRR28246669",1}' | bgzip > SRR28246669_fragments.bed.gz

bedtools bamtobed -i SRR28246669_Q30_sorted_rmdup.bam | awk -vOFS='\t' \
                                                            '{match($7, /(^[0-9]+)[SH]/, x)
                                                            lc=x[1]
                                                            match($7, /([0-9]+)[SH]$/, x) 
                                                            rc=x[1]
                                                            if (lc == "") lc=0
                                                            if (rc == "") rc=0
                                                            if (lc < fc && rc < fc) print $0}' \
                                                            fc=150 |\
                                       sort -k1,1 -k2,2n | gzip > SRR28246669.bed.gz
                                       
zcat SRR28246669.bed.gz | awk -vOFS='\t' '{print "chr"$0}' | bgzip > SRR28246669_fragments.bed.gz
tabix SRR28246669_fragments.bed.gz

cat \
<(zcat SRR28246669_fragments.bed.gz \
  | grep -E "^chr[0-9]|^chrX|^chrY" \
  | bedtools flank -i - -r 1 -l 0 -g hg38.chrom.sizes \
  | awk -vOFS='\t' '{$6="+"; print}') \
<(zcat SRR28246669_fragments.bed.gz \
  | grep -E "^chr[0-9]|^chrX|^chrY" \
  | bedtools flank -i - -r 0 -l 1 -g hg38.chrom.sizes \
  | awk -vOFS='\t' '{$6="-"; print}') \
> SRR28246669_fragments_flank.bed


zcat SRR28246669_fragments.bed.gz \
  | grep -E "^chr[0-9]|^chrX|^chrY" \
  | bedtools flank -i - -r 0 -l 1 -g hg38.chrom.sizes \
  | awk -vOFS='\t' '{$6="-"; print $1,$2,$3}' > SRR28246669_fragments_L.bed

zcat SRR28246669_fragments.bed.gz \
  | grep -E "^chr[0-9]|^chrX|^chrY" \
  | bedtools flank -i - -r 1 -l 0 -g hg38.chrom.sizes \
  | awk -vOFS='\t' '{$6="+"; print $1,$2,$3,$4}' > SRR28246669_fragments_R.bed

paste SRR28246669_fragments_L.bed SRR28246669_fragments_R.bed > SRR28246669_fragments_L_R.bedpe

awk '{if($1=="chr3" && $2>(196082123-100000) && $3<(196082123+100000)) print $0}' human_cCREs.bed | awk '{print $0 "\t" NR-1}' > ccre.bed  
zcat k562_hq_10_sorted.bed.gz | awk '{if($1=="3" && $2>(196082123-100000) && $3<(196082123+100000)) print "chr"$1 "\t" $2 "\t" $3 "\t" $4}' > fire.bed  
bedtools intersect -a ccre.bed -b fire.bed -wa -wb -loj | awk '{print $4 "\t" $8}' | uniq > ccre_reads.txt

import pandas as pd
import numpy as np
from itertools import combinations
import igraph as ig

ccre_read = pd.read_table('ccre_reads.txt', header=None)
ccre_read.columns = ['cre', 'read']
read_lst = ccre_read['read'].drop_duplicates().values
read_lst = np.delete(read_lst, np.where(read_lst=='.'))

cre_lst = ccre_read['cre'].drop_duplicates().values
m = np.zeros([len(cre_lst), len(cre_lst)])
m = m.astype(int)

ccre_read.drop_duplicates(inplace=True)

for i in read_lst:
    cre_pair_lst = [list(j) for j in list(combinations(ccre_read[ccre_read['read']==i]['cre'].values, 2))]
    cre_pair_row = [p[0] for p in cre_pair_lst]
    cre_pair_col = [p[1] for p in cre_pair_lst]
    m[cre_pair_row, cre_pair_col] += 1

m = m + m.T

## without edge label
g = ig.Graph.Weighted_Adjacency(m, mode='undirected')
g.vs["name"] = cre_lst
g.vs["label"] = g.vs["name"]
row_indices, col_indices = np.triu_indices(m.shape[0], k=1)
g.es["weight"] = [i for i in m[row_indices, col_indices] if i!=0]
visual_style = {'edge_width':g.es['weight']}
ig.plot(g, 'TFRC_network.pdf', bbox=(3000, 3000), **visual_style)






