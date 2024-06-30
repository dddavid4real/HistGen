DIR_TO_COORDS="path/to/patch/folder"
DATA_DIRECTORY="path/to/wsi/folder"
CSV_FILE_NAME="./dataset_csv/wsi-report-data_no_duplicate.csv"
FEATURES_DIRECTORY=$DIR_TO_COORDS
ext=".svs"
save_storage="yes"
root_dir="./extract_scripts/logs/WSI-Report_log_"

# models="resnet50"
# models="ctranspath"
# models="plip"
models="dinov2_vitl"

declare -A gpus
gpus["resnet50"]=0
gpus["resnet101"]=0
gpus["ctranspath"]=0
gpus["dinov2_vitl"]=0
gpus['plil']=0

datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 16 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir$model".txt" 2>&1 &

done