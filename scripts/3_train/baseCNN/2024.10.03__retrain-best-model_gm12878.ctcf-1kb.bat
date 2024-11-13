@echo off
setlocal enabledelayedexpansion

set out_folder=output/best/BaseCNN-gm12878.ctcf-1kb
set config=output/best/old/BaseCNN-gm12878.ctcf-1kb/configures.yaml
set input=data/train/gm12878_ctcf/1000bp.50ms.onehot/data.h5

python train.py ^
    --config !config! ^
    --input !input! ^
    --output_folder !out_folder! ^
    --device gpu ^
    --eval_freq 1 ^
    --pin_memory_train True 