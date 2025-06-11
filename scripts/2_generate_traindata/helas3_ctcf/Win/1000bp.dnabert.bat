@echo off
set name=helas3_ctcf

python data_helper.py -m dnabert -s 1000 --min_size 50 ^
                      --pos_files out_dir\%name%\pairs\%name%.clustered_interactions.both_dnase.bedpe ^
                      --neg_files out_dir\%name%\pairs\%name%.neg_pairs_5x.from_singleton_inter_tf_random.bedpe ^
                      --genome_file data\raw\hg19.fa ^
                      -n %name%_distance_matched ^
                      -o out_dir\%name%\train_data\1000bp.50ms.dnabert
