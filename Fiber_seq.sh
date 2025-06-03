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

## install samtools
conda install -c bioconda samtools
cd /fs/home/jiluzhang/softwares/miniconda3/envs/LRS/lib
ln -s libcrypto.so.1.1 libcrypto.so.1.0.0

## pbmm2: A minimap2 SMRT wrapper for PacBio data: native PacBio data in -> native PacBio BAM out.
## SMRT: single-molecule, real-time
conda install -c bioconda pbmm2

wget -c https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

zcat gm12878_fiberseq.fastq.gz | head -n 10000 > test.fastq
gzip test.fastq

pbmm2 index Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz Homo_sapiens.GRCh38.dna.chromosome.22.mmi --preset HiFi
pbmm2 align Homo_sapiens.GRCh38.dna.chromosome.22.mmi test.fastq.gz test.bam --preset HiFi --sort -j 4 -J 2
# -j,--num-threads           INT    Number of threads to use, 0 means autodetection. [0]
# -J,--sort-threads          INT    Number of threads used for sorting; 0 means 25% of -j, maximum 8. [0]

#---------------------intallation failed (by cargo)-------------------------------------------------#
# wget -c https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.2.0%2Bcu118.zip
# unzip libtorch-shared-with-deps-2.2.0+cu118.zip
# # add to .bashrc
# #export LIBTORCH_CXX11_ABI=0
# #export LIBTORCH=/fs/home/jiluzhang/LRS/gm12878_ng/libtorch
# #export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
# #export DYLD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# rustup update stable

# conda install -c bioconda perl perl-app-cpanminus
# cpan FindBin

# cargo install --all-features fibertools-rs
#-------------------------------------------------------------------------------------------------------#

mamba install -c conda-forge -c bioconda fibertools-rs  # not support GPU acceleration

ft fire test.bam test_fire.bam
ft fire --extract test_fire.bam fire.bed.gz
ft fire --extract --all test_fire.bam all.bed.gz








