@REM GM12878.ctcf 
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.08.25_cnn_hyper-test-final-gm12878_ks7.9-cr2.4.4-maxpool3_1000bp ^
    --names ^
        gm12878-final-1000bp ^
    --output output\experiment\2024.08.26__MCIENet_final\gm12878_ctcf-1000bp ^
    --box_group Hl Hs CHs ExtDrop ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Hl Hs CHs ExtDrop
