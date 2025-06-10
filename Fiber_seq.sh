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

## install bedtools
conda install bioconda::bedtools

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


## GSM4411222	K562 Fiber-seq (500 U Hia5) rep1 run1  [36 files]
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304357/DS76915_run1.ZMW0_4588045.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304358/DS76915_run1.ZMW139527343_144967719.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304359/DS76915_run1.ZMW144967724_150276627.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304360/DS76915_run1.ZMW150276628_155517675.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_1.bam DS76915_run1.ZMW0_4588045.subreads.bam.1 \
                                            DS76915_run1.ZMW139527343_144967719.subreads.bam.1 \
                                            DS76915_run1.ZMW144967724_150276627.subreads.bam.1 \
                                            DS76915_run1.ZMW150276628_155517675.subreads.bam.1
                                            
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304361/DS76915_run1.ZMW155517676_160500284.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304362/DS76915_run1.ZMW160500285_165479819.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304363/DS76915_run1.ZMW165479825_170525518.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304364/DS76915_run1.ZMW170525519_175439978.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_2.bam DS76915_run1.ZMW155517676_160500284.subreads.bam.1 \
                                            DS76915_run1.ZMW160500285_165479819.subreads.bam.1 \
                                            DS76915_run1.ZMW165479825_170525518.subreads.bam.1 \
                                            DS76915_run1.ZMW170525519_175439978.subreads.bam.1
                                            
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304365/DS76915_run1.ZMW175439980_180290639.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304366/DS76915_run1.ZMW180290641_180554589.subreads.bam.1 
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304367/DS76915_run1.ZMW18546864_23136760.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304368/DS76915_run1.ZMW100599331_106037253.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_3.bam DS76915_run1.ZMW175439980_180290639.subreads.bam.1 \
                                            DS76915_run1.ZMW180290641_180554589.subreads.bam.1 \
                                            DS76915_run1.ZMW18546864_23136760.subreads.bam.1 \
                                            DS76915_run1.ZMW100599331_106037253.subreads.bam.1
                                            
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304369/DS76915_run1.ZMW23136761_27984214.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304370/DS76915_run1.ZMW27984219_32899121.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304371/DS76915_run1.ZMW32899123_37816548.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304372/DS76915_run1.ZMW37816551_42928753.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_4.bam DS76915_run1.ZMW23136761_27984214.subreads.bam.1 \
                                            DS76915_run1.ZMW27984219_32899121.subreads.bam.1 \
                                            DS76915_run1.ZMW32899123_37816548.subreads.bam.1 \
                                            DS76915_run1.ZMW37816551_42928753.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304373/DS76915_run1.ZMW42928754_48105718.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304374/DS76915_run1.ZMW4588049_9176345.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304375/DS76915_run1.ZMW48105720_53283186.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304376/DS76915_run1.ZMW53283188_58525065.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_5.bam DS76915_run1.ZMW42928754_48105718.subreads.bam.1 \
                                            DS76915_run1.ZMW4588049_9176345.subreads.bam.1 \
                                            DS76915_run1.ZMW48105720_53283186.subreads.bam.1 \
                                            DS76915_run1.ZMW53283188_58525065.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304377/DS76915_run1.ZMW58525067_63767636.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304378/DS76915_run1.ZMW63767638_69009863.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304379/DS76915_run1.ZMW106037254_111543261.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304380/DS76915_run1.ZMW69009866_74253535.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_6.bam DS76915_run1.ZMW58525067_63767636.subreads.bam.1 \
                                            DS76915_run1.ZMW63767638_69009863.subreads.bam.1 \
                                            DS76915_run1.ZMW106037254_111543261.subreads.bam.1 \
                                            DS76915_run1.ZMW69009866_74253535.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304381/DS76915_run1.ZMW74253537_79497404.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304382/DS76915_run1.ZMW79497408_84740510.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304383/DS76915_run1.ZMW84740513_89982258.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304384/DS76915_run1.ZMW89982259_95291391.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_7.bam DS76915_run1.ZMW74253537_79497404.subreads.bam.1 \
                                            DS76915_run1.ZMW79497408_84740510.subreads.bam.1 \
                                            DS76915_run1.ZMW84740513_89982258.subreads.bam.1 \
                                            DS76915_run1.ZMW89982259_95291391.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304385/DS76915_run1.ZMW9176347_13829431.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304386/DS76915_run1.ZMW95291396_100599328.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304387/DS76915_run1.ZMW111543264_117179308.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304388/DS76915_run1.ZMW117179311_122815763.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_8.bam DS76915_run1.ZMW9176347_13829431.subreads.bam.1 \
                                            DS76915_run1.ZMW95291396_100599328.subreads.bam.1 \
                                            DS76915_run1.ZMW111543264_117179308.subreads.bam.1 \
                                            DS76915_run1.ZMW117179311_122815763.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304389/DS76915_run1.ZMW122815764_128450970.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304390/DS76915_run1.ZMW128450971_134021306.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304391/DS76915_run1.ZMW134021308_139527337.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304392/DS76915_run1.ZMW13829432_18546863.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run1_9.bam DS76915_run1.ZMW122815764_128450970.subreads.bam.1 \
                                            DS76915_run1.ZMW128450971_134021306.subreads.bam.1 \
                                            DS76915_run1.ZMW134021308_139527337.subreads.bam.1 \
                                            DS76915_run1.ZMW13829432_18546863.subreads.bam.1


## GSM4411223	K562 Fiber-seq (500 U Hia5) rep1 run2 [32 files]
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304393/DS76915_run2.ZMW100009546_105513461.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304394/DS76915_run2.ZMW138086402_143524054.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304395/DS76915_run2.ZMW143524055_148898671.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304396/DS76915_run2.ZMW148898675_154208538.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_1.bam DS76915_run2.ZMW100009546_105513461.subreads.bam.1 \
                                            DS76915_run2.ZMW138086402_143524054.subreads.bam.1 \
                                            DS76915_run2.ZMW143524055_148898671.subreads.bam.1 \
                                            DS76915_run2.ZMW148898675_154208538.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304397/DS76915_run2.ZMW154208540_159516274.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304398/DS76915_run2.ZMW159516275_164890484.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304399/DS76915_run2.ZMW164890489_170590940.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304400/DS76915_run2.ZMW170590942_176556862.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_2.bam DS76915_run2.ZMW154208540_159516274.subreads.bam.1 \
                                            DS76915_run2.ZMW159516275_164890484.subreads.bam.1 \
                                            DS76915_run2.ZMW164890489_170590940.subreads.bam.1 \
                                            DS76915_run2.ZMW170590942_176556862.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304401/DS76915_run2.ZMW176556865_180554585.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304402/DS76915_run2.ZMW18482772_24183322.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304403/DS76915_run2.ZMW24183323_29887193.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304404/DS76915_run2.ZMW105513462_110953881.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_3.bam DS76915_run2.ZMW176556865_180554585.subreads.bam.1 \
                                            DS76915_run2.ZMW18482772_24183322.subreads.bam.1 \
                                            DS76915_run2.ZMW24183323_29887193.subreads.bam.1 \
                                            DS76915_run2.ZMW105513462_110953881.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304405/DS76915_run2.ZMW29887196_35718918.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304406/DS76915_run2.ZMW35718922_41552461.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304407/DS76915_run2.ZMW41552462_47513677.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304408/DS76915_run2.ZMW47513682_53349027.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_4.bam DS76915_run2.ZMW29887196_35718918.subreads.bam.1 \
                                            DS76915_run2.ZMW35718922_41552461.subreads.bam.1 \
                                            DS76915_run2.ZMW41552462_47513677.subreads.bam.1 \
                                            DS76915_run2.ZMW47513682_53349027.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304409/DS76915_run2.ZMW53349030_59441810.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304410/DS76915_run2.ZMW59441811_65604203.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304411/DS76915_run2.ZMW65604208_71631799.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304412/DS76915_run2.ZMW6818240_12715107.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_5.bam DS76915_run2.ZMW53349030_59441810.subreads.bam.1 \
                                            DS76915_run2.ZMW59441811_65604203.subreads.bam.1 \
                                            DS76915_run2.ZMW65604208_71631799.subreads.bam.1 \
                                            DS76915_run2.ZMW6818240_12715107.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304413/DS76915_run2.ZMW71631801_77400528.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304414/DS76915_run2.ZMW77400529_83165637.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304415/DS76915_run2.ZMW10_6818239.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304416/DS76915_run2.ZMW83165639_88804092.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_6.bam DS76915_run2.ZMW71631801_77400528.subreads.bam.1 \
                                            DS76915_run2.ZMW77400529_83165637.subreads.bam.1 \
                                            DS76915_run2.ZMW10_6818239.subreads.bam.1 \
                                            DS76915_run2.ZMW83165639_88804092.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304417/DS76915_run2.ZMW88804097_94503336.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304418/DS76915_run2.ZMW94503337_100009545.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304419/DS76915_run2.ZMW110953883_116392454.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304420/DS76915_run2.ZMW116392456_121831554.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_7.bam DS76915_run2.ZMW88804097_94503336.subreads.bam.1 \
                                            DS76915_run2.ZMW94503337_100009545.subreads.bam.1 \
                                            DS76915_run2.ZMW110953883_116392454.subreads.bam.1 \
                                            DS76915_run2.ZMW116392456_121831554.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304421/DS76915_run2.ZMW121831555_127206734.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304422/DS76915_run2.ZMW12715110_18482771.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304423/DS76915_run2.ZMW127206736_132645796.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304424/DS76915_run2.ZMW132645797_138086400.subreads.bam.1
samtools merge -@ 4 -o k562_rep1_run2_8.bam DS76915_run2.ZMW121831555_127206734.subreads.bam.1 \
                                            DS76915_run2.ZMW12715110_18482771.subreads.bam.1 \
                                            DS76915_run2.ZMW127206736_132645796.subreads.bam.1 \
                                            DS76915_run2.ZMW132645797_138086400.subreads.bam.1


## GSM4411224	K562 Fiber-seq (500 U Hia5) rep2 [26 files]
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304425/DS76916.ZMW103942438_111019436.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304426/DS76916.ZMW160826387_167706813.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304427/DS76916.ZMW167706820_174589296.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304428/DS76916.ZMW174589298_180554587.subreads.bam.1
samtools merge -@ 4 -o k562_rep2_1.bam DS76916.ZMW103942438_111019436.subreads.bam.1 \
                                       DS76916.ZMW160826387_167706813.subreads.bam.1 \
                                       DS76916.ZMW167706820_174589296.subreads.bam.1 \
                                       DS76916.ZMW174589298_180554587.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304429/DS76916.ZMW1_7931776.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304430/DS76916.ZMW21629276_28445116.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304431/DS76916.ZMW28445132_35324206.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304432/DS76916.ZMW35324210_42206790.subreads.bam.1
samtools merge -@ 4 -o k562_rep2_2.bam DS76916.ZMW1_7931776.subreads.bam.1 \
                                       DS76916.ZMW21629276_28445116.subreads.bam.1 \
                                       DS76916.ZMW28445132_35324206.subreads.bam.1 \
                                       DS76916.ZMW35324210_42206790.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304433/DS76916.ZMW42206791_49153137.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304434/DS76916.ZMW49153143_56035198.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304435/DS76916.ZMW56035200_62916042.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304436/DS76916.ZMW111019438_118228349.subreads.bam.1
samtools merge -@ 4 -o k562_rep2_3.bam DS76916.ZMW42206791_49153137.subreads.bam.1 \
                                       DS76916.ZMW49153143_56035198.subreads.bam.1 \
                                       DS76916.ZMW56035200_62916042.subreads.bam.1 \
                                       DS76916.ZMW111019438_118228349.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304437/DS76916.ZMW62916044_69730699.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304438/DS76916.ZMW69730701_76417543.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304439/DS76916.ZMW76417544_83167207.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304440/DS76916.ZMW7931779_14943118.subreads.bam.1
samtools merge -@ 4 -o k562_rep2_4.bam DS76916.ZMW62916044_69730699.subreads.bam.1 \
                                       DS76916.ZMW69730701_76417543.subreads.bam.1 \
                                       DS76916.ZMW76417544_83167207.subreads.bam.1 \
                                       DS76916.ZMW7931779_14943118.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304441/DS76916.ZMW83167219_90046717.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304442/DS76916.ZMW90046719_96993846.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304443/DS76916.ZMW96993849_103942437.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304444/DS76916.ZMW118228351_125438227.subreads.bam.1
samtools merge -@ 4 -o k562_rep2_5.bam DS76916.ZMW83167219_90046717.subreads.bam.1 \
                                       DS76916.ZMW90046719_96993846.subreads.bam.1 \
                                       DS76916.ZMW96993849_103942437.subreads.bam.1 \
                                       DS76916.ZMW118228351_125438227.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304445/DS76916.ZMW125438230_132581806.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304446/DS76916.ZMW132581807_139856110.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304447/DS76916.ZMW139856111_147064353.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304448/DS76916.ZMW147064355_154075316.subreads.bam.1
samtools merge -@ 4 -o k562_rep2_6.bam DS76916.ZMW125438230_132581806.subreads.bam.1 \
                                       DS76916.ZMW132581807_139856110.subreads.bam.1 \
                                       DS76916.ZMW139856111_147064353.subreads.bam.1 \
                                       DS76916.ZMW147064355_154075316.subreads.bam.1

wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304449/DS76916.ZMW14943119_21629275.subreads.bam.1
wget -c https://sra-pub-src-2.s3.amazonaws.com/SRR11304450/DS76916.ZMW154075317_160826386.subreads.bam.1
samtools merge -@ 2 -o k562_rep2_7.bam DS76916.ZMW14943119_21629275.subreads.bam.1 \
                                       DS76916.ZMW154075317_160826386.subreads.bam.1

## download human reference fasta file
wget -c https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

## create index
pbmm2 index /fs/home/jiluzhang/LRS/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz GRCh38.mmi --preset CCS



#-----------------------------------------------------------------------------------------------#
## call_6mA_mapping
sample=$1

echo `date` $sample start

## subreads -> ccs
ccs -j 8 --hifi-kinetics $sample.bam $sample\_ccs.bam
echo `date` "  " $sample ccs done

## modify header
samtools view -H $sample\_ccs.bam > $sample\_ccs_header.sam
sed -i 's/101-717-300/101-789-500/g' $sample\_ccs_header.sam 
samtools reheader $sample\_ccs_header.sam $sample\_ccs.bam > $sample\_ccs_reheader.bam
echo `date` "  " $sample rehead done

## call 6mA
ft predict-m6a -t 8 -b 8 -k $sample\_ccs_reheader.bam $sample\_m6a.bam  # need to set --force-min-ml-score to filtering low quality sites
echo `date` "  " $sample 6mA_calling done

## mapping
pbmm2 align GRCh38.mmi $sample\_m6a.bam $sample\_m6a_aligned.bam --preset CCS --sort -j 8 -J 8
echo `date` "  " $sample mapping done

## remove tmp files
rm $sample\_ccs_header.sam
echo `date` "  " $sample all done
#-----------------------------------------------------------------------------------------------#

nohup ./k562_rep1_run1.sh > k562_rep1_run1.log &   # 938992
./call_6mA_mapping k562_rep1_run1_1
./call_6mA_mapping k562_rep1_run1_2
./call_6mA_mapping k562_rep1_run1_3
./call_6mA_mapping k562_rep1_run1_4
./call_6mA_mapping k562_rep1_run1_5
./call_6mA_mapping k562_rep1_run1_6
./call_6mA_mapping k562_rep1_run1_7
./call_6mA_mapping k562_rep1_run1_8
./call_6mA_mapping k562_rep1_run1_9

nohup ./k562_rep1_run2.sh > k562_rep1_run2.log &   # 939175
./call_6mA_mapping k562_rep1_run2_1
./call_6mA_mapping k562_rep1_run2_2
./call_6mA_mapping k562_rep1_run2_3
./call_6mA_mapping k562_rep1_run2_4
./call_6mA_mapping k562_rep1_run2_5
./call_6mA_mapping k562_rep1_run2_6
./call_6mA_mapping k562_rep1_run2_7
./call_6mA_mapping k562_rep1_run2_8

nohup ./k562_rep2.sh > k562_rep2.log &   # 940605
./call_6mA_mapping k562_rep2_1
./call_6mA_mapping k562_rep2_2
./call_6mA_mapping k562_rep2_3
./call_6mA_mapping k562_rep2_4
./call_6mA_mapping k562_rep2_5
./call_6mA_mapping k562_rep2_6
./call_6mA_mapping k562_rep2_7


#-----------------------------------------------------------------------------------------------#
## call_6mA_mapping_fire
sample=$1

echo `date` $sample start

## subreads -> ccs
#ccs -j 8 --hifi-kinetics $sample.bam $sample\_ccs.bam
#echo `date` "  " $sample ccs done

## modify header
samtools view -H $sample\_ccs.bam > $sample\_ccs_header.sam
sed -i 's/101-717-300/101-789-500/g' $sample\_ccs_header.sam 
samtools reheader $sample\_ccs_header.sam $sample\_ccs.bam > $sample\_ccs_reheader.bam
echo `date` "  " $sample rehead done

## call 6mA
ft predict-m6a -t 8 -b 8 -k --force-min-ml-score 244 $sample\_ccs_reheader.bam $sample\_m6a.bam
echo `date` "  " $sample 6mA_calling done

## mapping
pbmm2 align GRCh38.mmi $sample\_m6a.bam $sample\_m6a_aligned.bam --preset CCS --sort -j 8 -J 8
echo `date` "  " $sample mapping done

## call FIRE
ft fire -t 8 --width-bin 20 --bin-num 9 --best-window-size 50 --min-msp-length-for-positive-fire-call 85 $sample\_m6a_aligned.bam $sample\_fire.bam
echo `date` "  " $sample fire_calling done

## extract 6mA & FIRE
ft extract $sample\_fire.bam --m6a $sample\_m6a.bed.gz
ft fire --extract $sample\_fire.bam $sample\_fire.bed.gz
zcat $sample\_fire.bed.gz | awk '{if($5<=10) print$0}' > $sample\_fire_hq_10.bed
echo `date` "  " $sample 6mA_fire_extraction done

## remove tmp files
rm $sample\_ccs_header.sam $sample\_ccs_reheader.bam $sample\_m6a_aligned.bam $sample\_m6a.bam $sample\_m6a_aligned.bam.bai $sample\_fire.bed.gz
echo `date` "  " $sample all done
#-----------------------------------------------------------------------------------------------#

## rep1_run1
nohup ./k562_rep1_run1.sh > k562_rep1_run1.log &  # 2138343

## rep1_run2
nohup ./k562_rep1_run2.sh > k562_rep1_run2.log &  # 2143272

## rep2
nohup ./k562_rep2.sh > k562_rep2.log &  # 2142526






conda create -n Graph python=3.8
conda activate Graph

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-igraph
conda install -c conda-forge pycairo
conda install ipython

## https://www.aidoczh.com/igraph/generation.html




awk '{if($1=="chr13" && $2>25164000 && $3<25176000) print $0}' human_cCREs.bed | awk '{print $0 "\t" "CRE_"NR}' > ccre.bed  # 23
zcat k562_00_to_11_fire.bed.gz | awk '{if($1=="13" && $2>25164000 && $3<25176000) print "chr"$1 "\t" $2 "\t" $3 "\t" $4}' > fire.bed  # 108
bedtools intersect -a ccre.bed -b fire.bed -wa -wb | head
bedtools intersect -a ccre.bed -b fire.bed -wa -wb -loj | awk '{print $4 "\t" $8}' | uniq > ccre_fire.txt

import pandas as pd
import numpy as np
import igraph as ig

ccre_fire = pd.read_table('ccre_fire.txt', header=None)
ccre_fire.columns = ['cre', 'fire']
fire_lst = ccre_fire['fire'].drop_duplicates().values

cre_lst = ccre_fire['cre'].drop_duplicates().values
m = pd.DataFrame(np.zeros([len(cre_lst), len(cre_lst)]))
m = m.astype(int)
m.index = cre_lst
m.columns = cre_lst

for i in range(len(fire_lst)):
    if fire_lst[i]!='.':
        for idx in ccre_fire[ccre_fire['fire']==fire_lst[i]]['cre'].values:
            for col in ccre_fire[ccre_fire['fire']==fire_lst[i]]['cre'].values:
                if idx!=col:
                    m.loc[idx, col] += 1
m = np.array(m)
# m[m<3] = 0 
g = ig.Graph.Weighted_Adjacency(m, mode='undirected')
g.vs["name"] = cre_lst
g.vs["label"] = g.vs["name"]
row_indices, col_indices = np.triu_indices(m.shape[0], k=1)
g.es["weight"] = [i for i in m[row_indices, col_indices] if i!=0]
visual_style = {'edge_width':g.es['weight'],
                'edge_label':g.es['weight']}
ig.plot(g, 'test_graph.pdf', bbox=(3000, 3000), **visual_style)








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

## https://ccs.how/
# ccs -j 8 --hifi-kinetics test_raw.bam test_ccs.bam
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
# samtools view -H test_ccs.bam > test_ccs_header.sam
# 101-789-500
# 101-820-500
# 101-894-200
# 102-194-200
# 102-194-100
# 102-739-100
# 103-426-500
# 101-717-300 -> 101-789-500
# samtools reheader test_ccs_header.sam test_ccs.bam > test_ccs_reheader.bam

# ft predict-m6a -t 4 -b 2 -k test_ccs_reheader.bam test_m6a.bam
# -t, --threads <THREADS>  Threads [default: 8]
# -b, --batch-size <BATCH_SIZE>                  Number of reads to include in batch prediction [default: 1]
# -k, --keep                                                     Keep hifi kinetics data

# ft fire test_m6a.bam test_fire.bam

# pbmm2 index /fs/home/jiluzhang/LRS/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz GRCh38.mmi --preset CCS
# pbmm2 align GRCh38.mmi test_fire.bam test_fire_aligned.bam --preset CCS --sort -j 4 -J 2
# # -j,--num-threads           INT    Number of threads to use, 0 means autodetection. [0]
# # -J,--sort-threads          INT    Number of threads used for sorting; 0 means 25% of -j, maximum 8. [0]

# ft fire --extract test_fire.bam fire.bed.gz
# ft fire --extract --all test_fire.bam all.bed.gz

# ft extract test_fire_aligned.bam --m6a m6a.bed.gz
# ft extract test_fire_aligned.bam --msp msp.bed.gz

## PacBio BAM format specification: https://pacbiofileformats.readthedocs.io/en/latest/BAM.html#use-of-read-tags-for-per-read-base-base-modifications
