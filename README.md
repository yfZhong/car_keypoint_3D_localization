# Car Keypoint 3D Localization
Car Keypoint 3D Localization project provide methods for 3D car localization base on multi view cameras by 2D car detection and car wheel keypoints detection. 

<!-- TOC -->
- [Environment](#Environment)
- [Parameters](#Parameters)
- [Examples](#Examples)
<!-- /TOC -->

## Environment
### Build the docker 
```shell
~$ bash docker/build_docker.sh
```

## Parameters
#### The entry main python script is `lib/pred_imgs.py` 
| Parameters            | Require |             Notes                                                                                                                    |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| --store               | Yes      | Store name |
| --date                | Yes      | the date to processed |
| --data_root           | Yes      | Input data root |
| --model_root          | Yes      | model root |
| --camera_info_dir     | Yes      | Camera calibration information root|
| --save_root           | Yes      | folder for saving results |
| --use_gpu             | Yes      | whether to use gpu or cpu |
| --download_model      | Yes      | whether need to download model from hdfs |
| --download_image      | Yes      |whether need to download image from hdfs |
| --config-file-box     | Yes      | Path to car bbox detection model config file      |
| --config-file-kp      | Yes      | Path to car keypoint detection model config file               |
| --confidence-threshold| No       | Confidence threshold for car bbox output                   |
| --min-image-size      | No       | Smallest size of the image to feed to the box model                     |
| --min-image-size-kp   | No       | Smallest size of the image to feed to the keypoint model.                               |
| --show-mask-heatmaps  | No       | Show a heatmap probability for the top masks-per-dim masks                   |
| --masks-per-dim       | Yes      | Number of heatmaps per dimension to show                     |
| --img_root            | Yes      | Path to the input images
| --program_time_list   | Yes      | Path to the file with Store and data. each line with `${brand}\t${city}\t${store}\t${date}\t${time}`               |
| --save_root           | Yes      | Path for saving the output files 
| --vis_preds           | No       | whether to visulaze the preditions or not

## Examples

```shell
# 
~$ bash run_car_localization.sh \
 --store NANLING/guangzhou/qybc \
 --date 20201216 \
 --start_time 130000 \
 --data_root /root/code/car_localization/hdfs_data \
 --model_root /root/code/car_localization/models \
 --camera_info_dir /root/code/car_localization/CameraInfos \
 --use_gpu 1 \
 --download_model 0 \
 --download_image 1 \
 --upload_to_hdfs 1\
 --hdfs_pose_root /bj/prod/*** 
```
Main steps：
- Prepare
```shell
# prepare models and input images
```
- Run prediction
```shell
~$ # run pred
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
```
- Upload the final outputs
```shell
~$   bash upload.sh \
    --data_root ${DATA_ROOT} \
    --store ${STORE} \
    --hdfs_root ${HDFS_POSE_ROOT} \
    --date ${DATE}
```
