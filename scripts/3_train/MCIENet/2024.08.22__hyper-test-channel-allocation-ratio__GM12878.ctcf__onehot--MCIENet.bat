@echo off
setlocal enabledelayedexpansion

@REM cr1.8.1、cr2.6.2、cr3.4.3 | cr4.4.2、cr2.4.4

set name=cr1.8.1
set input_folder=data/train/gm12878_ctcf
set out_folder=output/gm12878_ctcf/2024.08.22_cnn_hyper-test-%name%_1000bp
set config_folder=conf/MCIENet_cr
set input_format=onehot

@REM Fix
set "lr=0.001"
set "bp_range_ls=1000"

@REM hyper
set "extractor_total_channels=100, 200, 300, 400"

set "classifier_dropout=0.5"
set "classifier_hidden_layer=3"
set "classifier_hidden_size=100, 200, 300"

set total=12
set current_test=0

echo out_folder
for %%a in (%bp_range_ls%) do (
    for %%b in (%classifier_hidden_layer%) do (
        for %%c in (%classifier_hidden_size%) do (
            for %%d in (%extractor_total_channels%) do (
                for %%e in (%classifier_dropout%) do (
                        set config=%config_folder%/%name%_%%abp.yaml
                        set input=%input_folder%/%%abp.50ms.%input_format%/data.h5
                        set out=%out_folder%/Bp%%a_Hl%%b_Hs%%c_CHs%%d_ClfDrop%%e
                        echo.
                        set /a current_test+=1
                        echo ">>>>>>>>>>>>>>>>> Task <<<<<<<<<<<<<<<<<"
                        echo Progress: !current_test! / %total%
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
                            --train.learning_rate %lr% ^
                            --model.extractor_total_channels %%d ^
                            --model.extractor_output_dim %%c ^
                            --model.classifier_input_dim %%c ^
                            --model.classifier_hidden_size %%c ^
                            --model.classifier_hidden_layer %%b ^
                            --model.classifier_dropout %%e ^
                    )
                )
            )
        )
    )


@REM python code\helper_scripts\sort_exp_result.py --folder %out_folder%

endlocal