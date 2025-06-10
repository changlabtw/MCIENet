@REM hela3-ctcf
python MCIENets\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.05.11_cnn_hyper-test-use-bn ^
        output\gm12878_ctcf\2024.05.11_cnn_hyper-test-no-bn ^
    --names use-bn no-bn ^
    --output output\experiment\2024.05.11__basecnn_BN-test.v2\gm12878_ctcf ^
    --folders_gp_name Bn ^
    --box_group CHs Bp Hs Hl ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Bp CHs Hs Hl Bn