#!/usr/bin/env python
"""
Projection-related functions.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import cv2
import numpy as np
import torch
import math
import polygon
from shapely.geometry import Polygon,Point

size=(2560, 1440)

def map_3dkpss_to_2dgroundimg(camera_cfg, ptss3d):
    ptss2d = []
    for i in range(len(ptss3d)):
    # for pts3d in ptss3d:
        pts3d=ptss3d[i]
        pts2d = map_3dkps_to_2dgroundimg(camera_cfg, pts3d)
        ptss2d.append(pts2d)
    return ptss2d

def map_3dkps_to_2dgroundimg(camera_cfg, pts3d):
    pts2d = []
    for pt3d in pts3d:
        pt2d = project_3dflat_to_2dground(camera_cfg.global_camera, pt3d)
        pt2d.append(pt3d[2])
        pts2d.append(pt2d)
    return pts2d

def map_pred_kps_to_3d(preds, camera_cfg, ch_name):
    '''

    :param preds: pred boxlist
    :param camera_cfg: camera config
    :param ch_name: channel name
    :return: project pred bbox keypoints to 3d plane
    '''

    boxes = preds.bbox
    labels = preds.get_field("labels")
    keypointss = preds.get_field("keypoints")
    kpss = keypointss.keypoints
    scores = keypointss.get_field("logits")
    kpss = torch.cat((kpss[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

    ###########
    ##TODO reorder
    inds = [0,1,2,3]

    kps_3dflatss = []
    box_scores = []
    for box_id in range(len(boxes)):
        anno = {}
        x1, y1, x2, y2 = boxes[box_id].to(torch.int).tolist()
        #x1, y1, w, h = boxes[box_id].to(torch.int).tolist()
        area = (x2-x1) * (y2 * y1)
        center_h = (y2+y1)/2.0

        kps = kpss[box_id]
        kps_clockwise = [kps[j] for j in inds]
        kps = np.array(kps_clockwise)

        all_0 = True
        kps_3dflats = []

        for i in range(kps.shape[0]):
            pt3dflat = project_2d_to_3dflat(camera_cfg.cameras[ch_name], kps[i][:2])
            # score
            pt3dflat[2] = kps[i][2]
            kps_3dflats.append(pt3dflat)

        #weights of this bbox
        polygon_score = math.sqrt(area/(size[0]*size[1])) * (size[1]-center_h)/size[1]

        kps_3dflatss.append(kps_3dflats)
        box_scores.append(polygon_score)

    return kps_3dflatss, box_scores

def add_3d_keypoints(preds, camera_cfg, ch_name, use_gts):
    '''

    :param preds: pred boxlist
    :param camera_cfg: camera config
    :param ch_name: channel name
    :return: project pred keypoints to 3d plane
    '''
    if preds == None:
        return None
    boxes = preds.bbox
    keypointss = preds.get_field("keypoints")
    kpss = keypointss.keypoints
    scores = keypointss.get_field("logits")
    kpss = torch.cat((kpss[:, :, 0:2], scores[:, :, None]), dim=2).numpy()

    kps_3dflatss = []
    boxes_xywh = []
    for box_id in range(len(boxes)):
        x1, y1, x2, y2 = boxes[box_id].to(torch.int).tolist()
        box_xywh = [x1, y1, x2 - x1 , y2 - y1]

        kps = np.array(kpss[box_id])

        # box area weight
        box_area_w = math.sqrt((x2-x1) * (y2-y1)/(size[0] * size[1]))

        # box position weight
        img_center = np.array([size[0]/2.0, size[1]/2.0])
        if (x1<img_center[0] and x2>img_center[0] and x1<img_center[0] and x2>img_center[0]):
            dist_to_center_w=1
        else:
            p = Point(size[0]/2, size[1]/2)
            poly = Polygon([(x1,y1), (x2,y1),(x2,y2),(x1,y2)])
            dist = p.distance(poly)
            dist_to_center_w = 1 - dist/(0.5*np.linalg.norm(np.array(size)))

        dist_to_border_w = max(np.min(np.array([x1, y1, size[0]-x2, size[1]-y2])/(0.5*(size[0]+size[1]))), 0.001)

        box_pts = np.array([(x1,y1), (x2,y1),(x2,y2),(x1,y2)])
        kps_in_box_area_pct = polygon.get_box_in_mask_pct(kps, box_pts)
        # weight = math.pow(box_area_w * kps_in_box_area_pct, 1/2)

        # weight = math.pow(box_area_w * dist_to_center_w * dist_to_border_w, 1 / 3)
        weight = (box_area_w + dist_to_center_w  + dist_to_border_w + kps_in_box_area_pct*2)/4.0


        kps_3dflats = []
        for i in range(kps.shape[0]):
            pt3dflat = project_2d_to_3dflat(camera_cfg.cameras[ch_name], kps[i][:2])
            # pt3dflat, prob = project_2d_to_3dflat(camera_cfg.cameras[ch_name], kps[i][:2], with_prob=True)
            # score
            pt3dflat[2] = kps[i][2]
            # for gt
            if use_gts == "True":
                if kps[i][2]==2: #invisible
                    pt3dflat[2] = 0.01
                elif kps[i][2]==1: #visible
                    pt3dflat[2] = 1
            else:
                # when the keypoint prediction weight is not reliable
                pt3dflat[2] = 1.0

            pt3dflat[2] = weight*pt3dflat[2]

            kps_3dflats.append(pt3dflat)

        kps_3dflatss.append(kps_3dflats)
        boxes_xywh.append(box_xywh)

    preds.add_field('keypoints_3d', torch.tensor(kps_3dflatss, dtype=torch.float, device='cpu'))
    idxs = [i for i in range(len(kps_3dflatss))]
    preds.add_field('idxs', torch.tensor(idxs, dtype=torch.int, device='cpu'))
    return preds

def project_3dmap_to_2d(camera, global_camera, pt_3dmap):
    point_3dmap = np.array([pt_3dmap[0], pt_3dmap[1], 1])
    ptflat = np.dot(global_camera.affine_mat2, point_3dmap)
    pt3d = np.array([ptflat[0], ptflat[1], 0])
    pts2d = camera.cam.project_3d_to_2d([pt3d])
    # point_2d = camera.cam.distort_point(pts2d[0])
    point_2d = pts2d[0]
    return point_2d


def project_2d_to_3dflat(camera, pt_2d, should_undistort=True, with_prob=False):
    point = np.array(pt_2d, dtype=float)
    pt3dflat = camera.cam.project_3d_given_z(point, 0, should_undistort=should_undistort, with_prob=with_prob)
    return pt3dflat

def project_2ds_to_3dflat(camera, pt_2ds):
    pt3dflats = []
    for pt_2d in pt_2ds:
        pt3dflat = project_2d_to_3dflat(camera, pt_2d)
        pt3dflats.append(pt3dflat)
    return pt3dflats

def project_3dflat_to_2dground(camera, pt_3dflat):
    input = [pt_3dflat[0], pt_3dflat[1], 1]
    pt2d = np.dot(camera.H_3to2map, input)
    pt2d = [int(pt2d[0]), int(pt2d[1])]
    return pt2d

def project_2dground_to_3dflat(camera, pt_2dground):
    input = [pt_2dground[0], pt_2dground[1], 1]
    pt2d = np.dot(camera.H_2mapto3, input)
    pt3dflat = [int(pt2d[0]), int(pt2d[1]), 0]
    return pt3dflat

### use perspective H ####
def project_2d_to_3dmap(camera, pt_2d):
    point = np.array([[pt_2d]], dtype=float)
    point_3dmap = cv2.perspectiveTransform(point, camera.H)
    pt_3dmap = [int(point_3dmap[0][0][0]), int(point_3dmap[0][0][1])]

    return pt_3dmap

def project_lot_2d_to_3dmap(camera, lot_2ds):
    lot_3dmaps = {}
    for lot_name, lot_2d in lot_2ds.iteritems():
        num_point = len(lot_2d)
        lot_3dmaps[lot_name] = []
        for point_id in range(num_point):
            point = np.array(lot_2d[point_id])
            pt3dflat = camera.cam.project_3d_given_z(point, 0, should_undistort=True)

            # input = np.array([[pt3dflat[0 : 2]]], dtype=np.float32)
            # pt2d = cv2.perspectiveTransform(input, camera.H)

            input = [pt3dflat[0], pt3dflat[1], 1]
            pt2d = np.dot(camera.H_3to2map, input)
            pt2d = [int(pt2d[0]), int(pt2d[1])]
            lot_3dmaps[lot_name].append(pt2d)
    return lot_3dmaps

def project_3d_bbox_back_to_2d(ress, camera_cfg):
    images = []

    for res in ress:
        img_path = res['img_path']
        preds = res['preds']
        boxes = preds.bbox.to(torch.int64).tolist()
        ids = preds.get_field("idxs").to(torch.int).tolist()

        image = {}
        image['image_path'] = img_path
        image['bbox'] = boxes
        image['ids'] = ids
        ch_name=res['ch_name']

        pose_3d = res['pose_3d']
        images.append(image)
        bboxes_from_3d=[]
        # for id in ids:
        #     if id<=0:
        #         continue
        #     box_3d=pose_3d[id-1]
        #     box_3d=np.array(box_3d)
        #     box_3d[:, 2] = 0.0
        #     bbox_from_3d = camera_cfg.cameras[ch_name].cam.project_3d_to_2d(box_3d)
        #     bboxes_from_3d.append(bbox_from_3d)
        for pose in pose_3d:
            pose = np.array(pose)
            pose[:, 2] = 0.0
            bbox_from_3d = camera_cfg.cameras[ch_name].cam.project_3d_to_2d(pose)
            at_least_one_pt_far_array = False
            w, h = 2560, 1440
            for pt in bbox_from_3d:
                if pt[0]< -(0.5*w) or pt[0]>(1.5*w) or pt[1]<-(0.5*h) or pt[1]>(1.5*h):
                    at_least_one_pt_far_array = True
                    break
            if at_least_one_pt_far_array:
                continue
            bboxes_from_3d.append(bbox_from_3d)

        preds.add_field('bbox_from_3d', torch.tensor(bboxes_from_3d, dtype=torch.float, device='cpu'))
    return ress

def project_kps3d_back_to_2d(ress, camera_cfg):
    for res in ress:
        preds = res['preds']
        projected_kpss = preds.get_field('keypoints_3d').to(torch.float).tolist()
        ch_name = res['ch_name']
        kpss_from_3d = []
        for kps in projected_kpss:
            kps_tmp = np.array(kps)
            kps_tmp[:, 2] = 0.0
            kps_from_3d = camera_cfg.cameras[ch_name].cam.project_3d_to_2d(kps_tmp)
            at_least_one_pt_far_array=False
            w, h = 2560, 1440
            for pt in kps_from_3d:
                if pt[0]< -(0.5*w) or pt[0]>(1.5*w) or pt[1]>-(0.5*h) or pt[1]>(1.5*h):
                    at_least_one_pt_far_array = True
                    break
            if at_least_one_pt_far_array:
                continue
            w = [0,0,0,0]
            for i in range(len(kps)):
                if kps[i][2]>0:
                    w[i] = 1
            kps_from_3d = np.concatenate([np.array(kps_from_3d), np.array(w).reshape((4, 1))], axis=1)

            kpss_from_3d.append(kps_from_3d)
        preds.add_field('kps_from_3d', torch.tensor(kpss_from_3d, dtype=torch.float, device='cpu'))
    return ress