@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\helas3_ctcf\2024.07.07_cnn_hyper-test-basic-structure ^
    --names basic-structure ^
    --output output\experiment\2024.06.27__basecnn_basic-structure\helas3-ctcf_dropout-and-BN ^
    --box_group ExtDrop ClfDrop ExtBn ClfBn ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group ExtDrop ClfDrop ExtBn ClfBn
