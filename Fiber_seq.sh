conda create -n LRS python=3.8
conda activate LRS

## Aspera installation (https://www.jianshu.com/p/ff826d6591f5)
wget https://download.asperasoft.com/download/sw/connect/3.10.0/ibm-aspera-connect-3.10.0.180973-linux-g2.12-64.tar.gz
tar zxvf ibm-aspera-connect-3.10.0.180973-linux-g2.12-64.tar.gz
bash ibm-aspera-connect-3.10.0.180973-linux-g2.12-64.sh
echo "export PATH=\$PATH:/data/home/jiluzhang/.aspera/connect/bin" >> ~/.bashrc
source ~/.bashrc

## install samtools
conda install -c bioconda samtools
cd /fs/home/jiluzhang/softwares/miniconda3/envs/LRS/lib
ln -s libcrypto.so.1.1 libcrypto.so.1.0.0

## pbmm2: A minimap2 SMRT wrapper for PacBio data: native PacBio data in -> native PacBio BAM out.
## SMRT: single-molecule, real-time
conda install -c bioconda pbmm2

## install fibertools
conda install -c conda-forge -c bioconda fibertools-rs  # not support GPU acceleration

## install jasmine & pbccs
## Call select base modifications in PacBio HiFi reads
conda install -c bioconda pbjasmine
conda install -c bioconda pbccs

# ## GM12878 Fiber-seq data: https://www.ebi.ac.uk/ena/browser/view/SRR29438436?show=reads (Nature Genetics, 2025)
# ascp -QT -l 300m -P 33001 -k 1 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh \
# era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/SRR294/036/SRR29438436/SRR29438436_subreads.fastq.gz gm12878_fiberseq.fastq.gz

## K562 Fiber-seq data: https://www.ebi.ac.uk/ena/browser/view/PRJNA612474 (Science, 2020)
## GSE146941: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146941
# ascp -QT -l 300m -P 33001 -k 1 -i ~/.aspera/connect/etc/asperaweb_id_dsa.openssh \
# era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/SRR113/059/SRR11304359/SRR11304359_subreads.fastq.gz k562_test.fastq.gz

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304366/DS76915_run1.ZMW180290641_180554589.subreads.bam.1  # bam file (include kinetics info)

## download human reference fasta file
wget -c https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

## alignment
# pbmm2 index /fs/home/jiluzhang/LRS/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz GRCh38.mmi --preset CCS
# cp DS76915_run1.ZMW180290641_180554589.subreads.bam.1 test_raw.bam
# pbmm2 align GRCh38.mmi test_raw.bam test.bam --preset CCS --sort -j 4 -J 2
# # -j,--num-threads           INT    Number of threads to use, 0 means autodetection. [0]
# # -J,--sort-threads          INT    Number of threads used for sorting; 0 means 25% of -j, maximum 8. [0]

# ft predict-m6a -t 4 -b 2 -k test.bam test_m6a.bam
# # -t, --threads <THREADS>  Threads [default: 8]
# # -b, --batch-size <BATCH_SIZE>                  Number of reads to include in batch prediction [default: 1]
# # -k, --keep                                                     Keep hifi kinetics data
# BINDINGKIT is not consistent (change to jasmine)

cp raw_data/DS76915_run1.ZMW180290641_180554589.subreads.bam.1 test_raw.bam
# <movie_name>/<zmw>/<subread_number>

## https://ccs.how/
ccs -j 8 --hifi-kinetics test_raw.bam test_ccs.bam
# -j,--num-threads          INT    Number of threads to use, 0 means autodetection. [0]
# --all-kinetics                   Calculate mean pulse widths (PW) and interpulse durations (IPD) for every ZMW.
# --hifi-kinetics                  Calculate mean pulse widths (PW) and interpulse durations (IPD) for every HiFi read.
## output: test_ccs.bam  test_ccs.bam.pbi  test_ccs.ccs_report.txt  test_ccs.zmw_metrics.json.gz
# ZMWs pass filters        : 1601 (26.37%)

# ## https://github.com/PacificBiosciences/jasmine
# jasmine --keep-kinetics -j 8 test_ccs.bam test_m6a.bam  # seems can not call 6mA (MM:Z:C+m?)  (change to ipdSummary)
# # --keep-kinetics            Keep kinetics tracks 'fi', 'fp', 'fn', 'ri', 'rp' and 'rn'.
# # -j,--num-threads     INT   Number of threads to use, 0 means autodetection. [0]

## ipdSummary: https://github.com/PacificBiosciences/kineticsTools/blob/master/kineticsTools/ipdSummary.py
# not output bam file with MM and ML tags (change to fibertools)

## change BINDINGKIT to the closest one
samtools view -H test_ccs.bam > test_ccs_header.sam
# 101-789-500
# 101-820-500
# 101-894-200
# 102-194-200
# 102-194-100
# 102-739-100
# 103-426-500
# 101-717-300 -> 101-789-500
samtools reheader test_ccs_header.sam test_ccs.bam > test_ccs_reheader.bam

ft predict-m6a -t 4 -b 2 -k test_ccs_reheader.bam test_m6a.bam
# -t, --threads <THREADS>  Threads [default: 8]
# -b, --batch-size <BATCH_SIZE>                  Number of reads to include in batch prediction [default: 1]
# -k, --keep                                                     Keep hifi kinetics data

ft fire test_m6a.bam test_fire.bam

pbmm2 index /fs/home/jiluzhang/LRS/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz GRCh38.mmi --preset CCS
pbmm2 align GRCh38.mmi test_fire.bam test_fire_aligned.bam --preset CCS --sort -j 4 -J 2
# # -j,--num-threads           INT    Number of threads to use, 0 means autodetection. [0]
# # -J,--sort-threads          INT    Number of threads used for sorting; 0 means 25% of -j, maximum 8. [0]

ft fire --extract test_fire.bam fire.bed.gz
ft fire --extract --all test_fire.bam all.bed.gz

ft extract test_fire_aligned.bam --m6a m6a.bed.gz

## PacBio BAM format specification: https://pacbiofileformats.readthedocs.io/en/latest/BAM.html#use-of-read-tags-for-per-read-base-base-modifications


conda create -n Graph python=3.8
conda activate Graph

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-igraph
conda install -c conda-forge pycairo
conda install ipython


import igraph as ig

g = ig.Graph([(0,2), (1,2), (2,3)])
g.vs["name"] = ["E1", "E2", "E3", "P1"]
g.vs["label"] = g.vs["name"]
g.es["weight"] = [1, 2, 3]
ig.plot(g, 'test_graph.pdf', edge_width=g.es["weight"], bbox=(300, 300))

 





#---------------------fibertools with all features intallation failed (by cargo)-------------------------------------------------#
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
#---------------------------------------------------------------------------------------------------------------------------------#
