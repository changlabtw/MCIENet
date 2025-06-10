@echo off
setlocal enabledelayedexpansion

set config=conf/base-CNN.yaml
set input_folder=data/train/gm12878_ctcf
set out_folder=output/gm12878_ctcf/2024.06.27_cnn_hyper-test-basic-structure
set input_format=onehot

set "lr=0.001"
set "bp_range_ls=3000"

set "classifier_hidden_layer=3"
set "classifier_hidden_size=100"

set "extractor_hidden_size=10 100"
set "extractor_pool_type=max, avg"
set "extractor_feature_aggregation=fc, avgpool, maxpool"
set "extractor_feature_agg_rate=0, 0.2, 0.5, 0.8"

set "extractor_dropout=0.0"
set "classifier_dropout=0.5"

set "extractor_bn=False"
set "classifier_bn=False"

@REM set "extractor_dropout=0.0, 0.3, 0.5"
@REM set "classifier_dropout=0.0, 0.3, 0.5"

@REM set "extractor_bn=True, False"
@REM set "classifier_bn=True, False"

set total=48
set current_test=0

for %%a in (%lr%) do (
    for %%b in (%bp_range_ls%) do (
        for %%c in (%classifier_hidden_layer%) do (
            for %%d in (%classifier_hidden_size%) do (
                for %%e in (%extractor_hidden_size%) do (
                    for %%f in (%extractor_pool_type%) do (
                        for %%g in (%extractor_feature_aggregation%) do (
                            for %%h in (%extractor_feature_agg_rate%) do (
                                for %%i in (%extractor_dropout%) do (
                                    for %%j in (%classifier_dropout%) do (
                                        for %%k in (%extractor_bn%) do (
                                            for %%l in (%classifier_bn%) do (
                                                set config=conf/base-CNN_%%bbp.yaml
                                                set input=%input_folder%/%%bbp.50ms.%input_format%/data.h5
                                                set out=%out_folder%/Bp%%b_Lr%%a_CHs%%e_Hs%%d_Hl%%c_Pt%%f_Fa%%g_Far%%h_ExtDrop%%i_ClfDrop%%j_ExtBn%%k_ClfBn%%l
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
                                                    --train.learning_rate %%a ^
                                                    --model.extractor_hidden_size %%e ^
                                                    --model.extractor_output_dim %%d ^
                                                    --model.extractor_dropout %%i ^
                                                    --model.extractor_bn %%k
                                                    --model.extractor_pool_type %%f ^
                                                    --model.extractor_feature_aggregation %%g ^
                                                    --model.extractor_feature_agg_rate %%h ^
                                                    --model.classifier_input_dim %%d ^
                                                    --model.classifier_hidden_size %%d ^
                                                    --model.classifier_hidden_layer %%c ^
                                                    --model.classifier_dropout %%j ^
                                                    --model.classifier_bn %%l
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

@REM python code\helper_scripts\sort_exp_result.py --folder %out_folder%

endlocal