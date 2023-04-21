#!/bin/bash
mkdir -p data
cd data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ILPkXSnA8-cLTlPnEr_GFYLmVgM-EKrq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ILPkXSnA8-cLTlPnEr_GFYLmVgM-EKrq" -O in_hand_manipulation.zip && rm -rf /tmp/cookies.txt
unzip in_hand_manipulation.zip -d ./in_hand_manipulation
wget http://ptak.felk.cvut.cz/public_datasets/coin-tracking/ctr.tar.gz
mkdir coin_tracking
tar xzvf ctr.tar.gz -C ./coin_tracking
cd ..
