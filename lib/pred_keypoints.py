#!/usr/bin/env python
"""
Functions for keypoint predicting.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import torch
from maskrcnn_benchmark.structures.keypoint import CarKeypoints
from maskrcnn_benchmark.structures.bounding_box import BoxList
import boxlist_process
import numpy as np
import vis
import os
import os.path as osp
import cv2

crop_delta = 20

def get_boxes_kps(img, preds, kp_model, score_thr=0.99):
    boxes = preds.bbox.to(torch.int64).tolist()
    labels = preds.get_field("labels").to(torch.int).tolist()
    scores = preds.get_field("scores").to(torch.float).tolist()
    # box_height_thr = 200
    # box_width_thr = 200
    box_height_thr = 0
    box_width_thr = 0
    new_boxes = []
    new_labels=[]
    new_scores = []
    new_keypoints = []
    new_keypoints_scores = []

    for box_id in range(len(boxes)):
        x1, y1, x2, y2 = boxes[box_id]
        if (y2-y1)<box_height_thr or x2-x1<box_width_thr:
            continue
        box = boxes[box_id]
        score = scores[box_id]
        if score < score_thr:
            continue

        box = [round(box[0])-crop_delta, round(box[1])-crop_delta, round(box[2])+2*crop_delta, round(box[3])+2*crop_delta]
        box = boxlist_process.bbox_in_img(img, box)
        new_boxes.append(box)
        new_scores.append(scores[box_id])
        new_labels.append(labels[box_id])

        img_crop = img[box[1]:box[3], box[0]:box[2], :]
        composite, kp_preds = kp_model.run_on_opencv_image(img_crop)
        # boxes = preds.bbox.to(torch.int64).tolist()
        # labels = preds.get_field("labels").to(torch.int).tolist()
        keypointss = kp_preds.get_field("keypoints")
        kpss = keypointss.keypoints.to(torch.float32)
        kscores = keypointss.get_field("logits").to(torch.float32).tolist()
        offset_x = box[0]
        offset_y = box[1]
        x = kpss[..., 0] + offset_x
        y = kpss[..., 1] + offset_y

        box_kpss = np.concatenate(
            (x.reshape((-1,1)), y.reshape((-1,1)), kpss[...,2].reshape((-1,1))),
            axis=1).tolist()
        new_keypoints.append(box_kpss)
        new_keypoints_scores.append(kscores[0])

    if len(new_boxes) <=0:
        return None
    proposal = BoxList(torch.tensor(np.array(new_boxes), device='cuda'), preds.size, mode='xyxy')
    proposal.add_field('scores',  torch.tensor(np.array(new_scores), device='cuda'))
    proposal.add_field('labels',  torch.tensor(np.array(new_labels), device='cuda'))
    proposal.add_field('keypoints', CarKeypoints(torch.tensor(new_keypoints, device='cuda'), preds.size))
    proposal.extra_fields['keypoints'].add_field('logits', torch.tensor(np.array(new_keypoints_scores), device='cuda'))

    return proposal


def get_boxes_kps_pad(img, preds, kp_model,  score_thr=0.99):
    boxes = preds.bbox.to(torch.int64).tolist()
    labels = preds.get_field("labels").to(torch.int).tolist()
    scores = preds.get_field("scores").to(torch.float).tolist()
    # box_height_thr = 200
    # box_width_thr = 200
    box_height_thr = 0
    box_width_thr = 0
    new_boxes = []
    new_labels=[]
    new_scores = []
    new_keypoints = []
    new_keypoints_scores = []

    boxes=np.array(boxes)
    if (boxes.shape[0]>0):
        boxes_xywh = np.hstack((boxes[:,0].reshape(-1,1),
                                boxes[:,1].reshape(-1,1),
                                (boxes[:,2]-boxes[:,0]).reshape(-1,1),
                                (boxes[:,3]-boxes[:,1]).reshape(-1,1)))
        iou_matrix = compute_iou_matrix(boxes_xywh)

    for box_id in range(len(boxes)):
        x1, y1, x2, y2 = boxes[box_id]
        if (y2-y1)<box_height_thr or x2-x1<box_width_thr:
            continue
        box = boxes[box_id]
        score = scores[box_id]
        if score < score_thr:
            continue

        # box = [round(box[0])-crop_delta, round(box[1])-crop_delta, round(box[2])+2*crop_delta, round(box[3])+2*crop_delta]
        box = [round(box[0]), round(box[1]), round(box[2]), round(box[3])]
        box = boxlist_process.bbox_in_img(img, box)
        new_boxes.append(box)
        new_scores.append(scores[box_id])
        new_labels.append(labels[box_id])

        # img_crop = img[box[1]:box[3], box[0]:box[2], :]
        img_crop = get_occlusion_pad_crop(img, boxes_xywh, box_id, iou_matrix)

        h, w, c = img_crop.shape
        ## pad
        img_p = pad_img(img_crop, 1)
        # img_crop = img_p

        composite, kp_preds = kp_model.run_on_opencv_image(img_p)
        keypointss = kp_preds.get_field("keypoints")
        kpss = keypointss.keypoints.to(torch.float32)
        kscores = keypointss.get_field("logits").to(torch.float32).tolist()


        offset_x = box[0]
        offset_y = box[1]

        x = kpss[..., 0] - w + offset_x
        y = kpss[..., 1] - h + offset_y

        box_kpss = np.concatenate(
            (x.reshape((-1,1)), y.reshape((-1,1)), kpss[...,2].reshape((-1,1))),
            axis=1).tolist()

        new_keypoints.append(box_kpss)
        new_keypoints_scores.append(kscores[0])

    if len(new_boxes) <=0:
        return None
    proposal = BoxList(torch.tensor(np.array(new_boxes), device='cuda'), preds.size, mode='xyxy')
    proposal.add_field('scores',  torch.tensor(np.array(new_scores), device='cuda'))
    proposal.add_field('labels',  torch.tensor(np.array(new_labels), device='cuda'))
    proposal.add_field('keypoints', CarKeypoints(torch.tensor(new_keypoints, device='cuda'), preds.size))
    proposal.extra_fields['keypoints'].add_field('logits', torch.tensor(np.array(new_keypoints_scores), device='cuda'))

    return proposal

def pad_img(img, pad_scale):
    h, w, c = img.shape
    img_p = np.zeros((h*(2*pad_scale+1), w*(2*pad_scale+1), 3), np.uint8)
    img_p[:, :] = (0, 0, 0)
    img_p[pad_scale*h:((pad_scale+1))*h, pad_scale*w:(pad_scale+1)*w, :] = img
    return img_p

def compute_iou_matrix(boxes):
    iou_matrix = np.zeros((len(boxes), len(boxes)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # distance of two clusters
            cur_iou = box_iou(boxes[i], boxes[j])

            iou_matrix[i][j] = cur_iou
            iou_matrix[j][i] = cur_iou
    return iou_matrix

def box_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    s_sum = w1 * h1 + w2 * h2

    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    if left >= right or top >= bottom:
        return 0
    intersect = (right - left) * (bottom - top)

    return intersect / (s_sum - intersect)

def get_overlap_box(box_back, box_front):
    '''
    input: box_back: format: xywh
    input: box_front: format: xywh
    return: format: xyxy
    '''
    x1, y1, w1, h1 = box_back
    x2, y2, w2, h2 = box_front
    if y1>y2:
        return [x1, y1, 0, 0]
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = y2
    bottom = min(y1 + h1, y2 + h2)

    overlap_box = [left, top, right, bottom]
    return overlap_box

def get_occlusion_pad_crop(img, boxes, target_box_idx, iou_matrix):
    box = boxes[target_box_idx]

    box_xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    box_xyxy = [round(box_xyxy[0]), round(box_xyxy[1]), round(box_xyxy[2]), round(box_xyxy[3])]
    box_xyxy = boxlist_process.bbox_in_img(img, box_xyxy)
    img_crop = img[box_xyxy[1]:box_xyxy[3], box_xyxy[0]:box_xyxy[2], :].copy()

    ious = iou_matrix[target_box_idx].copy()
    max_index = np.argmax(ious)
    # choose the largest box that in front of current box
    while True:
        if ious[max_index] == 0:
            break
        if box[1] < boxes[max_index][1]:
            break
        ious[max_index] = 0
        max_index = np.argmax(ious)

    # pad part of the overlap box
    if ious[max_index] > 0:
        overlap_box = get_overlap_box(box, boxes[max_index])
        o_l, o_t, o_r, o_b = overlap_box

        # pad_box = [o_l, o_t, o_r, o_b ]
        if o_l == box[0] and o_r == (box[0] + box[2]):
            pad_box = [o_l, int(0.5 * (o_t + o_b)), o_r, o_b]
        elif o_l == box[0]:
            pad_box = [o_l, int(0.5 * (o_t + o_b)), int(0.5 * (o_l + o_r)), o_b]
        else:
            pad_box = [int(0.5 * (o_l + o_r)), int(0.5 * (o_t + o_b)), o_r, o_b]
        shift_pad_box = [pad_box[0] - box[0], pad_box[1] - box[1], pad_box[2] - box[0], pad_box[3] - box[1]]

        img_crop[shift_pad_box[1]:shift_pad_box[3], shift_pad_box[0]:shift_pad_box[2], :] = (0, 0, 0)
    return img_crop