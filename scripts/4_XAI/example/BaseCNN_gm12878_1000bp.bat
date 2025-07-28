python get_attr.py ^
    --model_folder "output/best/BaseCNN-gm12878.ctcf-1kb" ^
    --output_folder "output/XAI/BaseCNN-gm12878.ctcf-1kb" ^
    --data_folder "data/train/gm12878_ctcf/1000bp.50ms.onehot" ^
    --phases train val test ^
    --batch_size 500 ^
    --method "DeepLift" ^
    --crop_center 500 ^
    --crop_size 1000 ^
    --use_cuda True