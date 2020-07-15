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
| Parameters            | Required |             Notes                                                                                                                    |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| --config-file-box     | Yes      | Path to car bbox detection model config file                                                  |
| --config-file-kp      | Yes      | Path to car keypoint detection model config file                                                                          |
| --confidence-threshold| No       | Confidence threshold for car bbox output                                                                                                |
| --min-image-size      | No       | Smallest size of the image to feed to the box model                                                                                     |
| --min-image-size-kp   | No       | Smallest size of the image to feed to the keypoint model.                                                                                |
| --show-mask-heatmaps  | No       | Show a heatmap probability for the top masks-per-dim masks                   |
| --masks-per-dim       | Yes      | Number of heatmaps per dimension to show                                                                         |
| --img_root            | Yes      | Path to the input images
| --program_time_list   | Yes      | Path to the file with Store and data. each line with `${branc}\t${city}\t${store}\t${date}\t${time}`                                          |
| --save_root           | Yes      | Path for saving the output files 
| --vis_preds           | No       | whether to visulaze the preditions or not

## Examples
```shell
#  bash scripts/run_hdfs_data_pred.sh ${brand} ${city} ${store} ${date} ${time}
~$ bash scripts/run_hdfs_data_pred.sh NANLING guangzhou qybc 20200202 13:00
```
Main steps：
- Download data
```shell
~$ bash scripts/download.sh
```
- Run pred
```shell
~$ python lib/pred_imgs.py \
    --config-file-box ./maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml \
    --config-file-kp ./maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_rcnn_R_101_FPN_1x_caffe2_4s_shop_pad.yaml \
    --confidence-threshold 0.95 \
    --img_root hdfs_data \
    --program_time_list program_id_time.txt \
    --save_root hdfs_data \
    --vis_preds True
```
- Upload result
```shell
~$ bash scripts/upload.sh
```
