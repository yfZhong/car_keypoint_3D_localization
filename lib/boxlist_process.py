#!/usr/bin/env python
"""
Functions for processing boxlists.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import torch
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import CarKeypoints
import polygon
import file
from math import fabs

size=(2560, 1440)
box_in_mask_pct_thr = 0.1

def filter_preds(preds, box_height_thr=200, box_width_thr=200, area_thr=10):
    if preds==None:
        return None
    boxes = preds.bbox.to(torch.int64).tolist()
    labels = preds.get_field("labels").to(torch.int).tolist()
    scores = preds.get_field("scores").to(torch.float32).tolist()
    keypointss = preds.get_field("keypoints")
    kpss = keypointss.keypoints.to(torch.float32).tolist()
    logits =keypointss.get_field("logits").to(torch.float32).tolist()

    filter_boxes = []
    filter_keypoints = []
    filter_scores = []
    filter_logits = []
    filter_labels = []
    for box_id in range(len(boxes)):
        x1, y1, x2, y2 = boxes[box_id]
        if (y2-y1)<box_height_thr or x2-x1<box_width_thr:
            continue

        # filter boxes by kps
        kps = kpss[box_id].copy()
        kps = polygon.kps_to_parallelogram(kps)
        if kps is None:
            continue

        # kps = polygon.kps_in_bbox(kps, boxes[box_id])

        # filter boxes by kps area
        area = polygon.polygon_area_with_sign(kps)
        # remove small detections
        if fabs(area)<area_thr:
            continue

        # rm counter-clockwise pts
        if area > 0:
            continue

        ## remove down_left and down-right corner detections
        # if ((x2-x1)<0.5*size[0] or (y2-y1)<0.5*size[1])\
        #     and ((abs(x1-0)<50 and abs(y2-size[1])< 50) or ( abs(x2-size[0]) < 50 and abs(y2-size[1])< 50)):
        #     # and abs(y2 - size[1]) < 50:
        #     continue

        # at least one point unlabeled
        if kps[0][2]==0 or kps[1][2]==0 or kps[2][2]==0 or kps[3][2]==0:
            continue

        filter_boxes.append([x1, y1, x2, y2])
        filter_labels.append(labels[box_id])
        filter_scores.append(scores[box_id])

        filter_keypoints.append(kps)
        filter_logits.append(logits[box_id])
    if len(filter_boxes) == 0:
        return None
    boxlist = BoxList(filter_boxes, preds.size, preds.mode)
    boxlist.add_field('labels', torch.tensor(filter_labels, dtype=torch.int, device='cpu'))
    boxlist.add_field('scores', torch.tensor(filter_scores, dtype=torch.float32, device='cpu'))
    boxlist.add_field('keypoints', CarKeypoints(filter_keypoints, preds.size))
    boxlist.get_field("keypoints").add_field('logits', torch.tensor(filter_logits, dtype=torch.float32, device='cpu'))
    return boxlist


def filter_preds_by_mask(preds, mask):
    if preds==None:
        return None
    boxes = preds.bbox.to(torch.int64).tolist()
    labels = preds.get_field("labels").to(torch.int).tolist()
    scores = preds.get_field("scores").to(torch.float32).tolist()

    keypointss = preds.get_field("keypoints")
    kpss = keypointss.keypoints.to(torch.float32).tolist()
    logits = keypointss.get_field("logits").to(torch.float32).tolist()

    filter_boxes = []
    filter_labels = []
    filter_scores = []

    filter_keypoints = []
    filter_logits = []
    for box_id in range(len(boxes)):
        x1, y1, x2, y2 = boxes[box_id]
        box_pts = [[x1,y1],[x2, y1],[x2, y2],[x1, y2]]
        box_area_in_mask_pct = polygon.get_box_in_mask_pct(box_pts, mask)
        if box_area_in_mask_pct < box_in_mask_pct_thr:
            continue

        filter_boxes.append([x1, y1, x2, y2])
        filter_labels.append(labels[box_id])
        filter_scores.append(scores[box_id])

        filter_keypoints.append(kpss[box_id])
        filter_logits.append(logits[box_id])

    if len(filter_boxes) == 0:
        return None
    boxlist = BoxList(filter_boxes, preds.size, preds.mode)
    boxlist.add_field('labels', torch.tensor(filter_labels, dtype=torch.int, device='cpu'))
    boxlist.add_field('scores', torch.tensor(filter_scores, dtype=torch.float32, device='cpu'))
    boxlist.add_field('keypoints', CarKeypoints(filter_keypoints, preds.size))
    boxlist.get_field("keypoints").add_field('logits', torch.tensor(filter_logits, dtype=torch.float32, device='cpu'))
    return boxlist


def load_img_gts_boxlist(file_path):
    import json
    msg_dict = json.load(open(file_path, 'r'))

    images, annotations = msg_dict['images'], msg_dict['annotations']
    img_heights = {image['file_name']: image['height'] for image in images}
    img_widths = {image['file_name']: image['width'] for image in images}

    dic_img_annos = {}
    for anno in annotations:
        img_name = anno['image_id']
        if img_name not in dic_img_annos.keys():
            dic_img_annos[img_name] = []
        dic_img_annos[img_name].append(anno)

    dic_img_gts = {}
    for img_name, annos in dic_img_annos.items():
        boxes = []
        keypoints = []
        labels = []
        scores = []
        logits = []
        for anno in annos:
            box = anno['bbox']
            x1, y1, w, h = box[0], box[1], box[2], box[3]

            kps=np.array(anno['keypoints']).reshape(-1,3).tolist()
            keypoints.append(kps)
            boxes.append([x1, y1, x1+w, y1+h])
            labels.append(anno['category_id'])
            # score = (anno['bbox'][3] - anno['bbox'][1])*1.0/img_heights[img_name]
            # score = 1.0/2.0
            score = 1
            scores.append(score)
            # logits.append([1, 1, 1, 1])
            logits.append((1 * np.array([float(kps[0][2]), float(kps[1][2]), float(kps[2][2]), float(kps[3][2])])))
        size = (img_widths[img_name], img_heights[img_name])
        boxlist = BoxList(boxes, size, 'xyxy')
        boxlist.add_field('labels', torch.tensor(labels, dtype=torch.int, device='cpu'))
        boxlist.add_field('scores', torch.tensor(scores, dtype=torch.int, device='cpu'))
        boxlist.add_field('keypoints', CarKeypoints(keypoints, size))
        boxlist.get_field("keypoints").add_field('logits', torch.tensor(logits, dtype=torch.float32, device='cpu'))
        dic_img_gts[img_name] = boxlist
    return dic_img_gts
    # dic_img_kps={}
    # for anno in annotations:
    #     id = anno['id']
    #     img_id = anno['image_id']
    #     kps = anno['keypoints']
    #     box = anno['bbox']
    #     height = box[3]
    #     if height<300:
    #         continue
    #     if img_id not in dic_img_kps.keys():
    #         dic_img_kps[img_id] = []
    #     dic_img_kps[img_id].append(np.array(kps).reshape(-1,3).tolist())

# def save_box_list(preds):
#


    # return dic_img_kps

def bbox_in_img(img, box):
    H, W, _ = img.shape
    if box[0] < 0:
        box[0] = 0;
    if box[1] < 0:
        box[1]=0
    if box[2] > (W-1):
        box[2] = W-1
    if box[3] > (H-1):
        box[3] = (H-1)
    # if box[2] > W:
    #     box[2] = W
    # if box[3] > H:
    #     box[3] = H
    return box