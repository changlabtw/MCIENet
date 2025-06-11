@echo off
set name=helas3_ctcf
set in_folder=data\proc\%name%
set out_folder=data\train\%name%

python data_helper.py -m text -s 2000 --min_size 50 ^
                      --pos_files %in_folder%\%name%.clustered_interactions.both_dnase.bedpe ^
                      --neg_files %in_folder%\%name%.neg_pairs_5x.from_singleton_inter_tf_random.bedpe ^
                      --genome_file data\raw\hg19.fa ^
                      -o %out_folder%\2000bp.50ms.text
