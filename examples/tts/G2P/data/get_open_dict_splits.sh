#!/bin/bash

# download ipa dict splits from https://github.com/lingjzhu/CharsiuG2P/tree/main/data

cd /mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits && \
wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/data/dev/eng-us.tsv && \
mv eng-us.tsv dev_eng-us.tsv && \
wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/data/train/eng-us.tsv && \
mv eng-us.tsv train_eng-us.tsv && \
wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/data/test/eng-us.tsv && \
mv eng-us.tsv test_eng-us.tsv