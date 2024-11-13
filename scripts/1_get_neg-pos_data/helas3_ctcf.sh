#!/bin/bash
mkdir -p "out_dir/helas3_ctcf/pairs"
./preprocess/pipe.sh data/helas3_ctcf/TangZ_etal.Cell2015.ChIA-PET_HelaS3_CTCF.published_PET_clusters.no_black.txt \
                        data/helas3_ctcf/wgEncodeAwgDnaseUwdukeHelas3UniPk.narrowPeak \
                        data/helas3_ctcf/wgEncodeAwgTfbsBroadHelas3CtcfUniPk.narrowPeak \
                        helas3_ctcf \
                        out_dir/helas3_ctcf/pairs