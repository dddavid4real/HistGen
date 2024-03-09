model='histgen'
max_length=100
epochs=40
region_size=96
prototype_num=512

python main_train_AllinOne.py \
    --image_dir /path/to/feature \
    --ann_path /path/to/json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --num_layers 3 \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --lr_ve 1e-4 \
    --lr_ed 1e-4 \
    --step_size 3 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /path/to/storage \
    --step_size 1 \
    --gamma 0.8 \
    --seed 456789 \
    --log_period 1000 \
    --beam_size 3
