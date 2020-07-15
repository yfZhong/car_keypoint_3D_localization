#git pull origin master:master
#git submodule update --remote
. ./set_env.sh

DAY=$1
CUSTOMER=NANLING
CITY=guangzhou
STORE=qybc
HDFS_FOLDER=${hdfs_root}/customer/$CUSTOMER/$CITY/$STORE/videos/processed/body
TIME=110000
VIDEO_ROOT=./data/video
VIDEO_FOLDER=$VIDEO_ROOT/$DAY
CAMERAINFO_FOLDER=./CameraInfos/$CUSTOMER/$CITY/$STORE

if [ ! -d $VIDEO_FOLDER ]
then
    echo "mkdir $VIDEO_FOLDER"
    mkdir -p $VIDEO_FOLDER
    HDFS_VIDEO_PATH=$HDFS_FOLDER/${DAY}/*_${DAY}${TIME}.mp4.cut.mp4
    hdfs dfs -get $HDFS_VIDEO_PATH $VIDEO_FOLDER
fi

python lib/pred_videos2.py \
    --config-file-box ../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
    --config-file-kp ../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
    --confidence-threshold 0.999 \
    --customer $CUSTOMER \
    --city $CITY \
    --store $STORE \
    --time $DAY$TIME \
    --save_root data/$PATH 

