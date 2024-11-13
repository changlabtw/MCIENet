@REM gm12878_ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_ctcf\2024.04.27_cnn_hyper-test ^
        output\gm12878_ctcf\2024.04.29_cnn-pairs_hyper-test ^
    --names basecnn basecnn-pairs ^
    --output output\experiment\2024.04.14_basecnn_devide-vs-concat\gm12878_ctcf


@REM gm12878_polr2a
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\gm12878_polr2a\2024.04.14_cnn_hyper-test ^
        output\gm12878_polr2a\2024.04.24_cnn-pairs_hyper-test ^
    --names basecnn basecnn-pairs ^
    --output output\experiment\2024.04.14_basecnn_devide-vs-concat\gm12878_polr2a


@REM helas3_ctcf
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\helas3_ctcf\2024.04.16_cnn_hyper-test ^
        output\helas3_ctcf\2024.04.25_cnn-pairs_hyper-test ^
    --names basecnn basecnn-pairs ^
    --output output\experiment\2024.04.14_basecnn_devide-vs-concat\helas3_ctcf


@REM k562_polr2a
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\k562_polr2a\2024.04.16_cnn_hyper-test ^
        output\k562_polr2a\2024.04.25_cnn-pairs_hyper-test ^
    --names basecnn basecnn-pairs ^
    --output output\experiment\2024.04.14_basecnn_devide-vs-concat\k562_polr2a


@REM mcf7_polr2a
python MCIENet\helper_scripts\compare_exp_result.py ^
    --folders ^
        output\mcf7_polr2a\2024.04.16_cnn_hyper-test ^
        output\mcf7_polr2a\2024.04.26_cnn-pairs_hyper-test ^
    --names basecnn basecnn-pairs ^
    --output output\experiment\2024.04.14_basecnn_devide-vs-concat\mcf7_polr2a