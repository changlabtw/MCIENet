python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_polr2a\2024.05.10_cnn_hyper-test-use-bn ^
        output\gm12878_polr2a\2024.05.10_cnn_hyper-test-no-bn ^
    --names use-bn no-bn ^
    --output output\experiment\2024.05.07_basecnn_BN-test\gm12878_polr2a ^
    --box_group CHs Bp