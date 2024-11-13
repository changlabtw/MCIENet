CUDA_VISIBLE_DEVICES=0 python get_attr.py \
    --model_folder "output/best/MCIENet-helas3.ctcf-1kb" \
    --output_folder "output/XAI/MCIENet-helas3.ctcf-1kb" \
    --data_folder "data/train/helas3_ctcf/3000bp.50ms.onehot" \
    --phases train val test \
    --batch_size 300 \
    --method "DeepLift" \
    --crop_center 1500 \
    --crop_size 1000 \
    --use_cuda True
