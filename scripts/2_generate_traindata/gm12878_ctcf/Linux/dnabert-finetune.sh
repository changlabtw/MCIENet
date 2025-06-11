#!/bin/bash

folder=data/train/gm12878_ctcf

python code/helper_scripts/to_dnabert_fine-tuning_format.py \
    --file ${folder}/1000bp.50ms.text/data.h5 \
    --output_folder ${folder}/1000bp.50ms.dnabert-finetune
