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
