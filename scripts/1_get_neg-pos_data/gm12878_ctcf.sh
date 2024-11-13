#!/bin/bash
mkdir -p "out_dir/gm12878_ctcf/pairs"
./preprocess/pipe.sh data/gm12878_ctcf/TangZ_etal.Cell2015.ChIA-PET_GM12878_CTCF.published_PET_clusters.no_black.txt \
                        data/gm12878_ctcf/wgEncodeAwgDnaseUwdukeGm12878UniPk.narrowPeak \
                        data/gm12878_ctcf/wgEncodeAwgTfbsBroadGm12878CtcfUniPk.narrowPeak \
                        gm12878_ctcf \
                        out_dir/gm12878_ctcf/pairs