#!/usr/bin/env python
"""
Functions for calculating pose precision and recall.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import os,sys
import json
import _init_paths
import file
#reload(sys)
#sys.setdefaultencoding("utf-8")
import argparse
import numpy as np
import os.path as osp
import shapely
from shapely.geometry import Polygon, MultiPoint

IOU_TH = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description='process json.')
    parser.add_argument('--gt_pose_root', type=str, help='gt_pose_root')
    parser.add_argument('--det_pose_root', type=str, help='det_pose_root')
    parser.add_argument('--program_time_list', type=str, help='input list')
    # parser.add_argument('--output_json', type=str, help='output_json')
    args = parser.parse_args()
    return args

def polygon_iou(pts1, pts2):

    if len(pts1) == 0 or len(pts2) == 0:
        return 0

    poly1 = Polygon(pts1).convex_hull
    poly2 = Polygon(pts2).convex_hull

    if not poly1.intersects(poly2):
        return 0

    union_pts = np.concatenate((pts1, pts2))
    union_poly = MultiPoint(union_pts).convex_hull
    iou = 0
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = union_poly.area
        if union_area == 0:
            return 0
        iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
    return iou

def single_iou_matrix(gt_info, dt_info):
    num_gt, num_dt = len(gt_info), len(dt_info)
    iou_matrix = np.zeros((num_gt, num_dt))
    for gt_index, gt_demo in enumerate(gt_info):
        gt_pts = gt_demo
        for dt_index, dt_demo in enumerate(dt_info):
            dt_pts = dt_demo
            iou_matrix[gt_index, dt_index] = polygon_iou(gt_pts, dt_pts)
    return iou_matrix

def count_tp(iou_matrix, threshold):
    """
    :param iou_matrix:
    :param threshold:  iou_threshold
    :return:
    """
    rows, columns = iou_matrix.shape
    tp = 0
    is_stop = np.all(iou_matrix < threshold)
    while not is_stop:
        max_index = np.argmax(iou_matrix)
        row = max_index // columns
        column = max_index % columns
        iou_matrix[row, :] = 0
        iou_matrix[:, column] = 0
        is_stop = np.all(iou_matrix < threshold)
        tp += 1
    return tp

def compute_recall_and_precision(gt_results, dt_results):
    tp_all = 0
    gt_all = 0
    dt_all = 0

    for img_name in gt_results:
        gt_info = gt_results[img_name]['dets']
        if img_name in dt_results:
            dt_info = dt_results[img_name]['dets']
            iou_matrix = single_iou_matrix(gt_info, dt_info)
            tp = count_tp(iou_matrix, IOU_TH)
            tp_all += tp
            gt_all += len(gt_info)
            dt_all += len(dt_info)
    print(tp_all, dt_all, gt_all)
    if tp_all == 0 and len(dt_results) == 0:
        return 0, 0
    return tp_all / gt_all, tp_all / dt_all


def fomat_det(pose_root, pose_list):
    pose_list=file.load_lines(pose_list)
    annos={}
    for pose_line in pose_list:
        customer, city, store, day, time = pose_line.split("\t")
        # pose_path0 = osp.join(customer, city, store, "car", "pose", day, time[:6]+".json")
        # pose_path = osp.join(pose_root, pose_path0)
        # if not os.path.exists(pose_path):
        #     pose_path0 = osp.join(customer, city, store, "car", "pose", day, "pose.json")
        #     pose_path = osp.join(pose_root, pose_path0)
        # if not os.path.exists(pose_path):
        #     pose_path0 = osp.join(customer, city, store, "car", "pose", day, "130000.json")
        #     pose_path = osp.join(pose_root, pose_path0)
        pose_path0 = osp.join(customer, city, store, "car", "pose", day, "pose.json")
        pose_path = osp.join(pose_root, pose_path0)
        if not os.path.isfile(pose_path):
            annos[pose_path0] = {'dets':[], 'ids':[]}
            continue
        print(pose_path)
        poses=json.load(open(pose_path, 'r'))
        dets = []
        ids = []
        for pose in poses:
            id = pose['id']
            pose = pose['cords']
            pose = np.array(pose)[:,:2]
            dets.append(pose)
            ids.append(id)
        annos[pose_path0] = {'dets':dets, 'ids':ids}

    return annos



if __name__=='__main__':
    args = parse_args()
    gt = fomat_det(args.gt_pose_root, args.program_time_list)
    det = fomat_det(args.det_pose_root, args.pose_list)
    recall, precision = compute_recall_and_precision(gt, det)
    print("{}/{}".format(round(precision*100,2), round(recall*100,2)))