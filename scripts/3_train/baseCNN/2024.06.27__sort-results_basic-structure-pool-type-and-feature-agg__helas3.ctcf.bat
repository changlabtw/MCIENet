@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\helas3_ctcf\2024.06.27_cnn_hyper-test-basic-structure ^
    --names basic-structure ^
    --output output\experiment\2024.06.27__basecnn_basic-structure\helas3-ctcf_pool-type-and-feature-agg ^
    --box_group CHs Pt Fa Far ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group CHs Pt Fa Far
