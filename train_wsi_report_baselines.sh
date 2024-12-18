token_select='uniform_sampling'
token_num=256
model='r2gen'
max_length=100
epochs=30

python main_train_AllinOne.py \
    --image_dir /path/to/feature \
    --ann_path /path/to/json \
    --dataset_name wsi_report \
    --model_name $model \
    --token_select $token_select \
    --token_num $token_num \
    --max_seq_length $max_length \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --lr_ve 1e-4 \
    --lr_ed 1e-5 \
    --step_size 3 \
    --topk 32 \
    --save_dir /path/to/storage \
    --step_size 1 \
    --gamma 0.8 \
    --seed 456789 \
    --log_period 1000 \
    --beam_size 3