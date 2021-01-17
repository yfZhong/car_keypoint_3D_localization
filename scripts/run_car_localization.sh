#!/usr/bin/env bash

source argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--store', default="NANLING/guangzhou/qybc")
parser.add_argument('--date', default='20201001')
parser.add_argument('--start_time', default="130000")
parser.add_argument('--data_root', default='/root/code/car_localization/data')
parser.add_argument('--camera_info_dir', default='/root/code/car_localization/CameraInfos')
parser.add_argument('--model_root', default='/root/code/car_localization/models')
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--download_image', default=1, type=int)
parser.add_argument('--download_model', default=1, type=int)
parser.add_argument('--upload_to_hdfs', default=1, type=int)
parser.add_argument('--hdfs_pose_root', default="/bj/yfzhong")
EOF


[ -f "program_id_time.txt" ] && rm program_id_time.txt

#set env
#. ./set_env.sh
if [ ${DOWNLOAD_MODEL} = 1 ]; then
  MODEL_DIR=${MODEL_ROOT}/4s_shop
  if [ ! -d $MODEL_DIR ]
  then
      mkdir -p $MODEL_DIR
  fi
  if [ -d $MODEL_DIR/res101_kp ]
  then
      rm -rf $MODEL_DIR/res101_kp
      rm -rf $MODEL_DIR/res101_box
      rm -rf $MODEL_DIR/res101_pad_kp
  fi

  version='v1.0'
  hdfscli download /bj/yfzhong/car/4s_shop/models/${version}/res101_pad_kp $MODEL_DIR
  hdfscli download /bj/yfzhong/car/4s_shop/models/${version}/res101_box $MODEL_DIR

fi

local_fold=${DATA_ROOT}/${STORE}/car/images/${DATE}
if [ ${DOWNLOAD_IMAGE} = 1 ]; then
  # download image
  #bash download.sh --store ${STORE} --start_time ${START_TIMR} --data_dir ${DATA_ROOT} --date ${DATE}
  fold=/bj/yfzhong/${STORE}/car/images/${DATE}
  rm -rf ${local_fold}/* || true
  mkdir -p ${local_fold} || true

  echo hdfscli download ${fold}/_done_ ${local_fold}/
  hdfscli download ${fold}/_done_ ${local_fold} || true
  echo hdfscli download -f ${fold}/ch* ${local_fold}/
  hdfscli download -f ${fold}/ch* ${local_fold}/
  #echo ${customer}$'\t'${city}$'\t'${store}$'\t'${day}$'\t'${start_time} >>'program_id_time.txt'
  echo "Download data to : "$fold
fi

if [ ! -f ${local_fold}/_done_ ]; then
  echo "No input data done signal detected. Images have not been prepared!"
  exit
fi

find ${local_fold} -name "*.jpg" > image_list.csv
num_images=`cat image_list.csv | wc -l`
echo "Number of images: "${num_images}

if [ ${num_images} = 0 ]; then
  echo "No input image was found!"
  exit
fi
#STORE_TAB=$(echo ${STORE} |sed "s/\//\t/g")'

if [ ${USE_GPU} = 1 ]; then
  export  CUDA_VISIBLE_DEVICES=0
else
  export  CUDA_VISIBLE_DEVICES=""
fi
# run pred
python lib/pred_imgs.py \
    --config-file-box ../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
    --config-file-kp ../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_rcnn_R_101_FPN_1x_caffe2_4s_shop_pad.yaml \
    --confidence-threshold 0.95 \
    --img_root ${DATA_ROOT} \
    --camera_info_dir ${CAMERA_INFO_DIR} \
    --save_root ${DATA_ROOT} \
    --store ${STORE} \
    --date ${DATE} \
    --start_time ${START_TIME} \
    --vis_preds True \
    --model_root ${MODEL_ROOT}


if [ ${UPLOAD_TO_HDFS} = 1 ]; then
  # upload result
  bash upload.sh \
    --data_root ${DATA_ROOT} \
    --store ${STORE} \
    --hdfs_root ${HDFS_POSE_ROOT} \
    --date ${DATE}
fi

