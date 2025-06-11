#!/bin/bash

# Run the preprocessing pipeline with the following parameters:
scripts/1_get_neg-pos_data/preprocess/pipe.sh \
    data/raw/helas3_ctcf/TangZ_etal.Cell2015.ChIA-PET_HelaS3_CTCF.published_PET_clusters.no_black.txt \
    data/raw/helas3_ctcf/wgEncodeAwgDnaseUwdukeHelas3UniPk.narrowPeak \
    data/raw/helas3_ctcf/wgEncodeAwgTfbsBroadHelas3CtcfUniPk.narrowPeak \
    helas3_ctcf \
    data/proc/helas3_ctcf

# Parameter 1: INTERS - Interaction file in BEDPE format (filtered against blacklisted regions)
# Parameter 2: DNASE - Dnase/open chromatin regions in BED format
# Parameter 3: TFPEAKS - Transcription factor peaks for the ChIA-PET protein in BED format
# Parameter 4: NAME - Prefix/name for the sample/experiment
# Parameter 5: DATADIR - Location of the output directory