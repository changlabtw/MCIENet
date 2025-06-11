#!/bin/bash

in_folder=data/proc/gm12878_ctcf
out_folder=data/train/gm12878_ctcf

python data_helper.py -m onehot -s 2000 --min_size 50 \
                      --pos_files ${in_folder}/gm12878_ctcf.clustered_interactions.both_dnase.bedpe \
                      --neg_files ${in_folder}/gm12878_ctcf.neg_pairs_5x.from_singleton_inter_tf_random.bedpe \
                      --genome_file data/raw/hg19.fa \
                      -o ${out_folder}/2000bp.50ms.onehot
