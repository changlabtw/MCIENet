#!/bin/bash
python train.py \
    --config "conf/MCIENet/gm12878_1000bp_best.yaml" \
    --input "data/train/gm12878_ctcf/1000bp.50ms.onehot/data.h5" \
    --output_folder "output/test/MCIENet" \
    --device gpu \
    --eval_freq 1 \
    --pin_memory_train True \
    --use_state_dict True \
    --train.max_epoch 5