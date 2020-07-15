#!/usr/bin/env python
"""
define arguments .

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
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
    # parser.add_argument(
    #     "--img_list",
    #     help="image path list",
    #     default="./data/lists/val.txt",
    # )
    parser.add_argument(
        "--hdfs_video_root",
        help="hdfs video root",
        default="/hdfs_root/customer/NANLING/guangzhou/qybc/videos/processed/body/",
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
        "--program_time_list",
        help="customer info",
        default="customer_list.txt",
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
