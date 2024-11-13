@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.07.20_cnn_hyper-test-final ^
    --names basic-structure ^
    --output output\experiment\2024.07.20__basecnn_final\gm12878_ctcf-3000bp ^
    --box_group Hl Hs CHs ClfDrop ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Hl Hs CHs ClfDrop
