CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd yolox


folds=("fold1" "fold2" "fold3" "fullfit")

for fold in "${folds[@]}"; do
    echo
    echo "Fold $fold"
    echo
#     cat "/workspace/kaggle_rsna_abdominal/yolox/exps/rsna_v1_${fold}_2.py"
    python /workspace/YOLOX/tools/train.py -f "/workspace/kaggle_rsna_abdominal/yolox/exps/rsna_v1_${fold}_2.py" -d 8 -b 32 --fp16 -o -c /workspace/YOLOX/yolox_m.pth --cache
done