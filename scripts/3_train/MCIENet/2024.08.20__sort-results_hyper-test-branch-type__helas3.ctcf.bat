@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^ ^
    --folders ^
        output\helas3_ctcf\2024.08.20_cnn_hyper-test-ks-7.9_1000bp ^
        output\helas3_ctcf\2024.08.20_cnn_hyper-test-ks-7.9_avgpool3_1000bp ^
        output\helas3_ctcf\2024.08.20_cnn_hyper-test-ks-7.9_avgpool7_1000bp ^
        output\helas3_ctcf\2024.08.20_cnn_hyper-test-ks-7.9_maxpool3_1000bp ^
        output\helas3_ctcf\2024.08.20_cnn_hyper-test-ks-7.9_maxpool7_1000bp ^
        output\helas3_ctcf\2024.08.20_cnn_hyper-test-ks-7.9_conv1x1_1000bp ^
    --names ^
        ks-7.9 ^
        ks-7.9_avgpool3 ^
        ks-7.9_avgpool7 ^
        ks-7.9_maxpool3 ^
        ks-7.9_maxpool7 ^
        ks-7.9_conv1x1 ^
    --output output\experiment\2024.08.20__MCIENet_branch-type\helas3-ctcf-1000bp ^
    --box_group Hs CHs ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Name Hs CHs
