@echo off
setlocal enabledelayedexpansion

set config=conf/base-CNN.yaml
set input_folder=data/train/gm12878_ctcf
set out_folder=output/gm12878_ctcf/2024.05.17_cnn_hyper-test-kernel-size
set input_format=onehot

set "lr=0.001"
set "hidden_size=300"
set "extractor_hidden_size=10, 50, 100, 200"
set "hidden_layer=2"

set "conv_kernel_size_ls=3, 4, 5, 6, 7, 8, 9, 10"

set "bp_range_ls=1000, 2000, 3000"

for %%a in (%lr%) do (
    for %%b in (%hidden_size%) do (
        for %%c in (%hidden_layer%) do (
            for %%d in (%bp_range_ls%) do (
                for %%e in (%extractor_hidden_size%) do (
                    for %%f in (%conv_kernel_size_ls%) do (
                        set config=conf/base-CNN_%%dbp.yaml
                        set input=%input_folder%/%%dbp.50ms.%input_format%/data.h5
                        set out=%out_folder%/Lr%%a_CHs%%e_Hs%%b_Hl%%c_Bp%%d_Ks%%f
                        echo.
                        echo ">>>>>>>>>>>>>>>>> Task <<<<<<<<<<<<<<<<<"
                        echo config: !config!
                        echo input: !input!
                        echo out: !out!
                        echo "<<<<<<<<<<<<<<<<< Task >>>>>>>>>>>>>>>>>"
                        echo.
                        python train.py ^
                            --config !config! ^
                            --input !input! ^
                            --output_folder !out! ^
                            --device gpu ^
                            --eval_freq 1 ^
                            --pin_memory_train True ^
                            --train.learning_rate %%a ^
                            --model.extractor_hidden_size %%e ^
                            --model.extractor_output_dim %%b ^
                            --model.extractor_conv_kernel_size %%f ^
                            --model.classifier_input_dim %%b ^
                            --model.classifier_hidden_size %%b ^
                            --model.classifier_hidden_layer %%c
                    )
                )
            )
        )
    )
)

python code\helper_scripts\sort_exp_result.py --folder %out_folder%

endlocal