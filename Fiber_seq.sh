## Aspera installation (https://www.jianshu.com/p/ff826d6591f5)

conda create -n LRS python=3.8
conda activate LRS

wget https://download.asperasoft.com/download/sw/connect/3.10.0/ibm-aspera-connect-3.10.0.180973-linux-g2.12-64.tar.gz
tar zxvf ibm-aspera-connect-3.10.0.180973-linux-g2.12-64.tar.gz
bash ibm-aspera-connect-3.10.0.180973-linux-g2.12-64.sh
echo "export PATH=\$PATH:/data/home/jiluzhang/.aspera/connect/bin" >> ~/.bashrc
source ~/.bashrc


## GM12878 Fiber-seq data: https://www.ebi.ac.uk/ena/browser/view/SRR29438436?show=reads (Nature Genetics, 2025)
ascp -QT -l 300m -P 33001 -k 1 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh \
era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/SRR294/036/SRR29438436/SRR29438436_subreads.fastq.gz gm12878_fiberseq.fastq.gz
# 2817900

## K562 Fiber-seq data: https://www.ebi.ac.uk/ena/browser/view/PRJNA612474 (Science, 2020)
ascp -QT -l 300m -P 33001 -k 1 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh \
era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/SRR113/059/SRR11304359/SRR11304359_subreads.fastq.gz k562_test.fastq.gz

aligner: pbmm2 (minimap2)
