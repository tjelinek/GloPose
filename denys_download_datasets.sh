FOLDER=/cluster/scratch/denysr/dataset/
WEIGHTS=/cluster/home/denysr/scratch/tmp/
scp jelint19@ptak.felk.cvut.cz:/datagrid/personal/rozumden/360photo/360photo.zip $FOLDER/360photo/
unzip $FOLDER/360photo/360photo.zip -d $FOLDER/360photo/

scp jelint19@ptak.felk.cvut.cz:/mnt/datasets/votrgbd/votrgbd.zip $FOLDER/votrgbd/
unzip $FOLDER/votrgbd/votrgbd.zip -d  $FOLDER/votrgbd/

wget http://ptak.felk.cvut.cz/public_datasets/coin-tracking/ctr.tar.gz -P $FOLDER/coin/
tar xvf $FOLDER/coin/ctr.tar.gz -C $FOLDER/coin/
scp -r jelint19@ptak.felk.cvut.cz:/mnt/datasets/coin-tracking/results/CTRBase $FOLDER/coin/results/
scp -r jelint19@radon.felk.cvut.cz:/ssd/export/D3S-masks/CTR/* $FOLDER/coin/results/D3S/

wget http://data.vicos.si/alanl/d3s/SegmNet.pth.tar -O $WEIGHTS/SegmNet.pth.tar
tar xvf  $WEIGHTS/SegmNet.pth.tar -C  $WEIGHTS
wget https://www.dropbox.com/s/hnv51iwu4hn82rj/s2dnet_weights.pth?dl=0 -O $FOLDER/s2dnet_weights.pth

# Download optical flow parameters
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip -O $WEIGHTS/models.zip
unzip $WEIGHTS/models.zip -d  $WEIGHTS/raft_models


#?? $FOLDER/ostrack/SEcmnet_ep0040-c.pth.tar