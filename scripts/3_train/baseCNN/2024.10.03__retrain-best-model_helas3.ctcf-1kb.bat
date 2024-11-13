@echo off
setlocal enabledelayedexpansion

set out_folder=output/best/BaseCNN-helas3.ctcf-1kb
set config=output/best/old/BaseCNN-helas3.ctcf-1kb/configures.yaml
set input=data/train/helas3_ctcf/1000bp.50ms.onehot/data.h5

python train.py ^
    --config !config! ^
    --input !input! ^
    --output_folder !out_folder! ^
    --device gpu ^
    --eval_freq 1 ^
    --pin_memory_train True 