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


############################################################################################################################################
## ./read2frag SRR28246682

sample=$1

echo `date` "......." $sample  start

mkdir $sample

cp /fs/home/jiluzhang/LRS/scNanoATAC_seq2/$sample\_1.fastq.gz .

minimap2 --MD -a -x map-ont -t 8 GRCh38_ONT.mmi $sample\_1.fastq.gz > $sample.sam

samtools view -bS -q 30 -@ 8 $sample.sam > $sample\_Q30.bam 
samtools sort -@ 8 $sample\_Q30.bam -o $sample\_Q30_sorted.bam
samtools index -@ 8 $sample\_Q30_sorted.bam
samtools rmdup -s $sample\_Q30_sorted.bam $sample\_Q30_sorted_rmdup.bam
samtools index -@ 8 $sample\_Q30_sorted_rmdup.bam

bedtools bamtobed -i $sample\_Q30_sorted_rmdup.bam | awk -vOFS='\t' \
                                                         '{match($7, /(^[0-9]+)[SH]/, x)
                                                           lc=x[1]
                                                           match($7, /([0-9]+)[SH]$/, x) 
                                                           rc=x[1]
                                                           if (lc == "") lc=0
                                                           if (rc == "") rc=0
                                                           if (lc < fc && rc < fc) print $0}' \
                                                           fc=150 |\
                                                    sort -k1,1 -k2,2n | gzip > $sample.bed.gz                  
zcat $sample.bed.gz | awk -vOFS='\t' '{print "chr"$0}' | bgzip > $sample\_fragments.bed.gz
tabix $sample\_fragments.bed.gz

zcat $sample\_fragments.bed.gz \
  | grep -E "^chr[0-9]|^chrX|^chrY" \
  | bedtools flank -i - -r 0 -l 1 -g hg38.chrom.sizes \
  | awk -vOFS='\t' '{$6="-"; print $1,$2,$3}' > $sample\_fragments_L.bed
zcat $sample\_fragments.bed.gz \
  | grep -E "^chr[0-9]|^chrX|^chrY" \
  | bedtools flank -i - -r 1 -l 0 -g hg38.chrom.sizes \
  | awk -vOFS='\t' '{$6="+"; print $1,$2,$3,$4}' > $sample\_fragments_R.bed
paste $sample\_fragments_L.bed $sample\_fragments_R.bed > $sample\_fragments_L_R.bedpe

rm $sample\_1.fastq.gz $sample.sam $sample\_Q30.bam $sample\_Q30_sorted.bam $sample\_Q30_sorted.bam.bai $sample.bed.gz $sample\_fragments_L.bed $sample\_fragments_R.bed

mv $sample* $sample

echo `date` "......." $sample  done
############################################################################################################################################

## 10 GM12878 cells
# SRR28246682  SRR28246693  SRR28246697  SRR28246699  SRR28246704 SRR28246706  SRR28246709  SRR28246712  SRR28246714  SRR28246722
for srr_id in SRR28246682  SRR28246693  SRR28246697  SRR28246699  SRR28246704 SRR28246706  SRR28246709  SRR28246712  SRR28246714  SRR28246722;do
    ./read2frag $srr_id >> gm12878_10.log
done


## mouse embryo
wget -c https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz



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

















