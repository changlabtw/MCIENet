CUDA_VISIBLE_DEVICES=0 python get_attr.py \
    --model_folder "output/best/MCIENet-gm12878.ctcf-1kb" \
    --output_folder "output/XAI/MCIENet-gm12878.ctcf-1kb" \
    --data_folder "data/train/gm12878_ctcf/3000bp.50ms.onehot" \
    --phases train val test \
    --batch_size 300 \
    --method "DeepLift" \
    --crop_center 1500 \
    --crop_size 1000 \
    --use_cuda True
