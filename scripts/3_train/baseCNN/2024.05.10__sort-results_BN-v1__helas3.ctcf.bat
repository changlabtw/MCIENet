@REM hela3-ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\helas3_ctcf\2024.05.10_cnn_hyper-test-use-bn ^
        output\helas3_ctcf\2024.05.10_cnn_hyper-test-no-bn ^
    --names use-bn no-bn ^
    --output output\experiment\2024.05.07_basecnn_BN-test\helas3_ctcf ^
    --box_group CHs Bp