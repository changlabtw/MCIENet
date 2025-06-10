@echo off
rem difference bp range encode
setlocal enabledelayedexpansion

set config=conf/base-CNN.yaml
set input_folder=data/train/gm12878_ctcf
set out_folder=output/gm12878_ctcf/2024.04.14_cnn_hyper-test
set input_format=onehot

set "lr=0.001"
set "hidden_size=200, 300"
set "extractor_hidden_size=200, 300"
set "hidden_layer=1, 2"

set "bp_range_ls=1000 2000 3000"

for %%a in (%lr%) do (
    for %%b in (%hidden_size%) do (
        for %%c in (%hidden_layer%) do (
            for %%d in (%bp_range_ls%) do (
                for %%e in (%extractor_hidden_size%) do (
                python train.py ^
                    --config %config% ^
                    --input %input_folder%/%%dbp.50ms.%input_format%/data.h5 ^
                    --output_folder %out_folder%/Lr%%a_Hs%%b_Hl%%c_Bp%%d_CHs%%e ^
                    --device gpu ^
                    --eval_freq 1 ^
                    --pin_memory_train True ^
                    --train.learning_rate %%a ^
                    --data.anchor_size %%d ^
                    --model.classifier_hidden_size %%b ^
                    --model.extractor_hidden_size %%e ^
                    --model.classifier_hidden_layer_n %%c
                )
            )
        )
    )
)

python code\helper_scripts\sort_exp_result.py --folder %out_folder%

endlocal