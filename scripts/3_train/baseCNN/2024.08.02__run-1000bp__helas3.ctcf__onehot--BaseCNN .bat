@echo off
setlocal enabledelayedexpansion

set input_folder=data/train/helas3_ctcf
set out_folder=output/helas3_ctcf/2024.08.02_cnn_hyper-test-final
set input_format=onehot

@REM Fix
set "lr=0.001"
set "bp_range_ls=1000"

@REM Basic structure
set "extractor_pool_type=max"
set "extractor_feature_aggregation=fc"
set "extractor_feature_agg_rate=0.5"

set "extractor_dropout=0.3"

set "extractor_bn=False"
set "classifier_bn=False"

@REM hyper
set "extractor_hidden_size=100"

set "classifier_dropout=0.0"
set "classifier_hidden_layer=3"
set "classifier_hidden_size=100"

set total=1
set current_test=0


for %%a in (%bp_range_ls%) do (
    for %%b in (%classifier_hidden_layer%) do (
        for %%c in (%classifier_hidden_size%) do (
            for %%d in (%extractor_hidden_size%) do (
                for %%e in (%classifier_dropout%) do (
                        set config=conf/base-CNN_%%abp.yaml
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
                            --model.extractor_hidden_size %%d ^
                            --model.extractor_output_dim %%c ^
                            --model.extractor_dropout %extractor_dropout% ^
                            --model.extractor_bn %extractor_bn% ^
                            --model.extractor_pool_type %extractor_pool_type% ^
                            --model.extractor_feature_aggregation %extractor_feature_aggregation% ^
                            --model.extractor_feature_agg_rate %extractor_feature_agg_rate% ^
                            --model.classifier_input_dim %%c ^
                            --model.classifier_hidden_size %%c ^
                            --model.classifier_hidden_layer %%b ^
                            --model.classifier_dropout %%e ^
                            --model.classifier_bn %classifier_bn%
                    )
                )
            )
        )
    )


python code\helper_scripts\sort_exp_result.py --folder %out_folder%

endlocal