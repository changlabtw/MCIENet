set "base_pair=1000"

python train.py ^
    --config "conf\MCIENet-basic_%base_pair%bp.yaml" ^
    --input "data\train\gm12878_ctcf\%base_pair%bp.50ms.onehot\data.h5" ^
    --output_folder "output\test\MCIENet" ^
    --device gpu ^
    --eval_freq 1 ^
    --pin_memory_train True ^
    --use_state_dict True ^
    --train.max_epoch 5 ^
    --train.learning_rate 0.001 ^
    --model.extractor_pool_proj_chs_ls "0.05,0.05" ^
    --model.extractor_pool_proj_ks_ls "9,9" ^
    --model.extractor_pool_proj_dilation_ls "1,1" ^
    --model.extractor_pool_proj_type_ls "avgpool,maxpool"
