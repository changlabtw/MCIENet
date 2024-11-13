@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.05.17_cnn_hyper-test-kernel-size ^
    --names kernel-size-test ^
    --output output\experiment\2024.05.17__basecnn_kernel-size-test\gm12878_ctcf ^
    --folders_gp_name Name ^
    --box_group CHs Bp Ks ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Bp CHs Hs Hl Ks