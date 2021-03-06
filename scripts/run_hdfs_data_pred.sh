#!/usr/bin/env bash

# root fold
CURR_FOLD="$(dirname "$(readlink -f "$0")")"
ROOT_FOLD=${CURR_FOLD}/../

if [ $# -lt 3 ];
then
    echo "Illegal number of parameters!"
    exit
fi

#set env
. ${CURR_FOLD}/set_env.sh

if [ -f "program_id.txt" ]
then
    rm program_id.txt
fi

echo $1 $2 $3 $4 >"program_id.txt"

# download data
bash ${CURR_FOLD}/download.sh

# run pred
cd ${ROOT_FOLD}
python lib/pred_imgs.py \
    --config-file-box ../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
    --config-file-kp ../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_rcnn_R_101_FPN_1x_caffe2_4s_shop_pad.yaml \
    --confidence-threshold 0.95 \
    --img_root hdfs_data \
    --program_time_list program_id_time.txt \
    --save_root hdfs_data \
    --vis_preds True

# upload result
bash ${CURR_FOLD}/upload.sh

