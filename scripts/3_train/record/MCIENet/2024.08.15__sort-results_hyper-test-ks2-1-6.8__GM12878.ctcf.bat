@REM hela3-ctcf
python code\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.08.15_cnn_hyper-test-ks2-1-6.8_1000bp ^
    --names ks2-1-6.8 ^
    --output output\experiment\2024.08.15_MCIENet_kernel-size\gm12878_ctcf-1000bp_ks2-1-6.8 ^
    --box_group Hl Hs CHs ClfDrop ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Hl Hs CHs ClfDrop
