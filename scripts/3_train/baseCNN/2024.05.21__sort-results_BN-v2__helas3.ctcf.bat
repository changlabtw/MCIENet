@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\helas3_ctcf\2024.05.21_cnn_hyper-test-use-bn ^
        output\helas3_ctcf\2024.05.21_cnn_hyper-test-no-bn ^
    --names use-bn no-bn ^
    --output output\experiment\2024.05.11__basecnn_BN-test.v2\helas3_ctcf ^
    --folders_gp_name Bn ^
    --box_group CHs Bp Hs Hl ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Bp CHs Hs Hl Bn