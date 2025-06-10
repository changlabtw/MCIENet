@REM gm12878_ctcf
python MCIENets\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.05.07_cnn_hyper-test-use-bn ^
        output\gm12878_ctcf\2024.05.07_cnn_hyper-test-no-bn ^
    --names ^
        use-bn ^
        no-bn ^
    --output output\experiment\2024.05.07_basecnn_BN-test\gm12878_ctcf ^
    --box_group CHs Bp