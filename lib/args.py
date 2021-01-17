#!/usr/bin/env python
"""
define arguments .

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-11
"""

import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="PyTorch Object Detection")
    parser.add_argument(
        "--config-file-box",
        default="../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_101_FPN_1x_caffe2_4s_shop.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--config-file-kp",
        default="../maskrcnn-benchmark/configs/caffe2/e2e_keypoint_only_R_101_FPN_1x_caffe2_4s_shop.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model.",
    )
    parser.add_argument(
        "--min-image-size-kp",
        type=int,
        default=192,
        help="Smallest size of the image to feed to the model.",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--img_root",
        help="image root",
        default="./data/val/",
    )
    parser.add_argument(
        "--model_root",
        help="model root",
        default="./models",
    )
    parser.add_argument(
        "--camera_info_dir",
        help="Path to CameraInfos",
        default="./CameraInfos",
    )
    parser.add_argument(
        "--hdfs_video_root",
        help="hdfs video root",
        default="/prod/yfzhong/NANLING/guangzhou/qybc/videos/processed/body/",
    )
    parser.add_argument(
        "--video_root",
        help="root for load input video",
        default="data/video/",
    )
    parser.add_argument(
        "--gt_annos",
        help="image annotation list",
        default="./data/annotations/keypoints_val.json",
    )
    parser.add_argument(
        "--store",
        help="store name",
        default="NANLING/guangzhou/qybc",
    )
    parser.add_argument(
        "--date",
        help="date time",
        default="20201201",
    )
    parser.add_argument(
        "--start_time",
        help="image produced time",
        default="130000",
    )
    parser.add_argument(
        "--save_root",
        help="root for save output images",
        default="data/result",
    )
    parser.add_argument("--vis_gts", default="False", type=str, help="vis ground truth")
    parser.add_argument("--vis_preds", default="False", type=str, help="vis result of multiple cameras truth")
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args
