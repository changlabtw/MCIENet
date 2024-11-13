@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^ ^
    --folders ^
        output\helas3_ctcf\2024.08.22_cnn_hyper-test-cr1.8.1_1000bp ^
        output\helas3_ctcf\2024.08.22_cnn_hyper-test-cr2.4.4_1000bp ^
        output\helas3_ctcf\2024.08.22_cnn_hyper-test-cr2.6.2_1000bp ^
        output\helas3_ctcf\2024.08.22_cnn_hyper-test-cr3.4.3_1000bp ^
        output\helas3_ctcf\2024.08.22_cnn_hyper-test-cr4.4.2_1000bp ^
    --names ^
        cr1.8.1 ^
        cr2.4.4 ^
        cr2.6.2 ^
        cr3.4.3 ^
        cr4.4.2 ^
    --output output\experiment\2024.08.22__MCIENet_channel-allocation-ratio\helas3_ctcf-1000bp ^
    --box_group Hs CHs ^
    --save_metrics F1 Precision Recall auPRCs auROC matthews_corrcoef ^
    --sort_args_group Name Hs CHs