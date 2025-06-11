CUDA_VISIBLE_DEVICES=1 python get_attr.py \
    --model_folder "output/best/MCIENet-helas3.ctcf-1kb" \
    --output_folder "output/XAI/MCIENet-helas3.ctcf-1kb_right-shift-500" \
    --data_folder "data/train/helas3_ctcf/3000bp.50ms.onehot" \
    --phases train val test \
    --batch_size 300 \
    --method "DeepLift" \
    --crop_center 2000 \
    --crop_size 1000 \
    --use_cuda True
