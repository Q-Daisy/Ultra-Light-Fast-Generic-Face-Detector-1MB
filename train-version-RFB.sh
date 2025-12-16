#!/usr/bin/env bash
model_root_path="./models/train-version-RFB"
log_dir="$model_root_path/logs"
log="$log_dir/log"
mkdir -p "$log_dir"

python3 -u train.py \
  --pretrained_ssd \
  ./models/pretrained/version-RFB-320.pth \
  --datasets \
  ./data/age3_voc \
  --validation_dataset \
  ./data/age3_voc \
  --net \
  RFB \
  --num_epochs \
  200 \
  --milestones \
  "95,150" \
  --lr \
  1e-3 \
  --batch_size \
  24 \
  --input_size \
  320 \
  --checkpoint_folder \
  ${model_root_path} \
  --num_workers \
  4 \
  --log_dir \
  ${log_dir} \
  --cuda_index \
  0 \
  2>&1 | tee "$log"


python3 -u train.py --pretrained_ssd ./models/pretrained/version-RFB-320.pth --datasets ./data/age3_voc --validation_dataset ./data/age3_voc --net RFB --num_epochs 200 --milestones "95,150" --lr 1e-3 --batch_size 24 --input_size 320 --num_workers 16 --cuda_index 0 