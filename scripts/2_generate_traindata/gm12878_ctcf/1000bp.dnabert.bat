@REM directly using dnabert pretrain model to generate Feature vector as input
@REM the result of this method is worse than using onehot encoding, not used in benchmark

python data_helper.py -m dnabert -s 2000 --min_size 50 ^
                      --pos_files out_dir/gm12878_ctcf/pairs/gm12878_ctcf.clustered_interactions.both_dnase.bedpe ^
                      --neg_files out_dir/gm12878_ctcf/pairs/gm12878_ctcf.neg_pairs_5x.from_singleton_inter_tf_random.bedpe ^
                      --genome_file data/raw/hg19.fa ^
                      -n gm12878_ctcf_distance_matched ^
                      -o out_dir/gm12878_ctcf/train_data/2000bp.50ms.dnabert
@REM train 9.80it/s 237888 大約 6:44:35
@REM val 10.34it/s 27762 大約 44:44 
@REM test 10.06it/s 58836 大約 1:37:28  
@REM 共 324486 筆， 9:6:47
