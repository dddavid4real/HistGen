model='histgen'
max_length=100
epochs=40
region_size=96
prototype_num=512

python main_test_AllinOne.py \
    --image_dir /path/to/feature \
    --ann_path /path/to/json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --step_size 1 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /path/to/storage \
    --step_size 1 \
    --gamma 0.8 \
    --seed 42 \
    --log_period 1000 \
    --load /path/to/checkpoint \
    --beam_size 3
