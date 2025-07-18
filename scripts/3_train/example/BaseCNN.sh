#!/bin/bash

base_pair=1000

python train.py \
    --config "conf/experiment/BaseCNN/${base_pair}bp.yaml" \
    --input "data/train/gm12878_ctcf/${base_pair}bp.50ms.onehot/data.h5" \
    --output_folder "output/test/BaseCNN" \
    --device gpu \
    --eval_freq 1 \
    --pin_memory_train True \
    --use_state_dict True \
    --train.max_epoch 5