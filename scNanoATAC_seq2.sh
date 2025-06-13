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

## install minimap2 (version: 2.29-r1283)
conda install bioconda::minimap2

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

samtools view -bS -q 30 -@ 8 SRR28246669.sam > SRR28246669_q30.bam 








