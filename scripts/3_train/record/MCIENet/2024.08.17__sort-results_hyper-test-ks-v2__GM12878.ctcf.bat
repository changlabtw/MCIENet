@REM GM12878.ctcf 
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.08.17_cnn_hyper-test-ks1-9_1000bp ^
        output\gm12878_ctcf\2024.08.17_cnn_hyper-test-ks2-7.9_1000bp ^
        output\gm12878_ctcf\2024.08.17_cnn_hyper-test-ks3-5.7.9_1000bp ^
        output\gm12878_ctcf\2024.08.17_cnn_hyper-test-ks4-3.5.7.9_1000bp ^
    --names ^
        ks1-9 ^
        ks2-7.9 ^
        ks3-5.7.9 ^
        ks4-3.5.7.9 ^
    --output output\experiment\2024.08.17__MCIENet_kernel-size.v2\gm12878_ctcf-1000bp-ks ^
    --box_group Name Hs CHs ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Name Hs CHs
