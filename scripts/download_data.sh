#!/bin/bash
mkdir -p /mnt/personal/jelint19/data/
cd /mnt/personal/jelint19/data/ || exit
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ILPkXSnA8-cLTlPnEr_GFYLmVgM-EKrq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ILPkXSnA8-cLTlPnEr_GFYLmVgM-EKrq" -O in_hand_manipulation.zip && rm -rf /tmp/cookies.txt
unzip in_hand_manipulation.zip -d ./in_hand_manipulation
wget http://ptak.felk.cvut.cz/public_datasets/coin-tracking/ctr.tar.gz
mkdir coin_tracking
tar xzvf ctr.tar.gz -C ./coin_tracking
cd ..

mkdir behave
cd behave || exit
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date01.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date02.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date03.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date04.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date05.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date06.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date07.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/objects.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/calibs.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/split.json
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date01-02.tar
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date03.tar
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date04-06.tar
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/masks/masks-date07.tar
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/train_part1.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/train_part2.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/cvprw23/train_part3.zip
unzip "Date*.zip" -d sequences
