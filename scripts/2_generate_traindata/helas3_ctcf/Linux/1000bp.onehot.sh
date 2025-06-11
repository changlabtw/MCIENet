#!/bin/bash

name=helas3_ctcf
in_folder=data/proc/${name}
out_folder=data/train/${name}

python data_helper.py -m onehot -s 1000 --min_size 50 \
                      --pos_files ${in_folder}/${name}.clustered_interactions.both_dnase.bedpe \
                      --neg_files ${in_folder}/${name}.neg_pairs_5x.from_singleton_inter_tf_random.bedpe \
                      --genome_file data/raw/hg19.fa \
                      -o ${out_folder}/1000bp.50ms.onehot
