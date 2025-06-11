python train.py ^
    --config "conf\MCIENet_final\gm12878_ks7.9-cr2.4.4-maxpool3_1000bp.yaml" ^
    --input "data\train\gm12878_ctcf\1000bp.50ms.onehot\data.h5" ^
    --output_folder "output\test\MCIENet" ^
    --device gpu ^
    --eval_freq 1 ^
    --pin_memory_train True ^
    --use_state_dict True ^
    --train.max_epoch 5
