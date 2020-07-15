#!/usr/bin/env python
"""
Functions related to polygon calculation.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  #多边形
import cv2
import math
from math import fabs, sqrt, atan2
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import file

keypoint_score_thr = 0.0
keypoint_score_thr0 = 0.0
car_min_width = 1.6
car_max_height = 3.0
num_kps = 4
car_fix_length = 4.7
car_fix_width = 1.83

def kps_filter(pts, thr):
    filter_pts=[]
    for i in range(4):
        pt = pts[i]
        this_pt = pt[:2]
        if pt[2] > thr:
            filter_pts.append(this_pt)
    if len(filter_pts) == 0:
        # print("no pt left!")
        return []
    return filter_pts

def kps_convert(pts):
    filter_pts=[]
    for i in range(4):
        pt = pts[i]
        filter_pts.append([pt[0], pt[1]])
    return np.array(filter_pts)


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

def polygon_angle(pts1, pts2):
    line_a = (pts1[0] + pts1[1])*0.5 - (pts1[2] + pts1[3])*0.5
    line_b = (pts2[0] + pts2[1])*0.5 - (pts2[2] + pts2[3])*0.5
    angle = get_angle_pi(line_a, line_b)
    return radian_to_degree(angle)

def get_box_in_mask_pct(box_pts, mask):

    if len(box_pts) == 0 or len(mask) == 0:
        return 0

    poly1 = Polygon(box_pts).convex_hull
    # poly2 = Polygon(mask).convex_hull
    poly2 = Polygon(mask)

    if not poly1.intersects(poly2):
        return 0

    pct = 0
    try:
        inter_area = poly1.intersection(poly2).area
        box_area = poly1.area
        if box_area == 0:
            return 0
        pct = float(inter_area) / box_area
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
    return pct

def polygon_area(pts):
    """返回多边形面积
    """
    area = 0
    q = pts[-1]
    for p in pts:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return fabs(area / 2)

def polygon_area_with_sign(pts):
    """返回多边形面积
    """
    area = 0
    q = pts[-1]
    for p in pts:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2

def polygons_cluster(ptss, lid=[], iou_thr = 0.5):
    if len(ptss)==0:
        return {}, {}
    # split to two groups according num of pts.
    filter_ptss1 = []
    filter_ptss2 = []
    idxs_group = []
    idxs_group2 = []
    for i in range(len(ptss)):
        pts = ptss[i]
        filter_pts = kps_filter(pts, keypoint_score_thr)
        if len(filter_pts) > 2:
            filter_ptss1.append(np.array(filter_pts))
            idxs_group.append([i])
        else:
            if len(filter_pts)>0:
                idxs_group2.append([i])
                filter_ptss2.append(np.array(filter_pts))

    if len(filter_ptss1) == 0:
        return None
    # cluster group1 ptss
    keep_going = True
    idxs_group = np.array(idxs_group)
    while keep_going:
        keep_going = False
        if len(filter_ptss1)==1:
            break
        dists_M = np.ones((len(filter_ptss1), len(filter_ptss1)))
        for i in range(len(filter_ptss1)):
            for j in  range(i+1, len(filter_ptss1)):
                # distance of two clusters
                dist = 1 - polygon_iou(filter_ptss1[i], filter_ptss1[j])
                # punishment to large angle detections
                angle = polygon_angle(filter_ptss1[i], filter_ptss1[j])
                if angle>60:
                    dist=1
                dists_M[i][j] = dist
                dists_M[j][i] = dist

        # detections from the same channel should be in different clusters
        for i in range(len(filter_ptss1)):
            idxs1 = idxs_group[i]
            for j in  range(i+1, len(filter_ptss1)):
                idxs2 = idxs_group[j]
                has_box_from_same_img=False
                for id1 in idxs1:
                    for id2 in idxs2:
                        if lid[id1] == lid[id2]:
                            has_box_from_same_img=True
                if has_box_from_same_img:
                    dists_M[i][j] = 1.0
                    dists_M[j][i] = 1.0

        agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                      linkage='average', distance_threshold=(1-iou_thr))
        cluster_idxs = agg.fit_predict(dists_M)
        new_ptss = []
        new_idxs_group = []
        for c_id in range(np.max(cluster_idxs)+1):
            idxs = np.where(cluster_idxs == c_id)[0]
            if idxs.shape[0] >1:
                keep_going = True
            this_ptss = filter_ptss1[idxs[0]]
            this_idxs = idxs_group[idxs[0]]
            for i in range(1, len(idxs)):
                this_ptss = np.concatenate((this_ptss, filter_ptss1[idxs[i]]), axis=0)
                this_idxs = np.concatenate((this_idxs, idxs_group[idxs[i]]), axis=0)
            new_ptss.append(this_ptss)
            new_idxs_group.append(this_idxs)
        filter_ptss1 = new_ptss
        idxs_group = new_idxs_group

    ## assign single edge ptss (assign group 2 ptss to clusters)
    # idxs_group2 = np.array(idxs_group2)
    # for i in range(len(filter_ptss2)):
    #     pts = filter_ptss2[i]
    #     id = idxs_group2[i]
    #     merged = False
    #     for g_id in range(len(filter_ptss1)):
    #         g_pts = filter_ptss1[g_id]
    #         all_p_in_group = True
    #         for pt in pts:
    #             if not is_in_2d_polygon(pt, g_pts):
    #                 all_p_in_group = False
    #                 break
    #         if all_p_in_group:
    #             idxs_group[g_id].append(id)
    #             merged = True
    #             break

    # output
    group_pts ={}
    g_id = 0
    ptss =  np.array(ptss)
    idx_map = {}
    for g_ids in idxs_group:
        g_ptss = ptss[g_ids[0]].reshape((1, 4, 3))
        idx_map[g_ids[0]] = g_id
        for k in range(1, len(g_ids)):
            g_ptss = np.concatenate((g_ptss, ptss[g_ids[k]].reshape((1, 4, 3))), axis=0)
            idx_map[g_ids[k]] = g_id
        group_pts[g_id] = g_ptss.tolist()
        g_id +=1

    return group_pts, idx_map

def polygons_cluster2(ptss, iou_thr = 0.5):

    # split to two groups according num of pts.
    filter_ptss1 = []
    filter_ptss2 = []
    idxs_group = []
    idxs_group2 = []
    for i in range(len(ptss)):
        pts = ptss[i]
        filter_pts = np.array(kps_filter(pts, keypoint_score_thr))
        if filter_pts is None:
            continue
        if len(filter_pts) > 2:
            filter_ptss1.append(filter_pts)
            idxs_group.append([i])
        else:
            if len(filter_pts)>0:
                idxs_group2.append([i])
                filter_ptss2.append(filter_pts)

    if len(filter_ptss1) == 0:
        return None
    # cluster group1 ptss
    keep_going = True
    idxs_group = np.array(idxs_group)
    while keep_going:
        keep_going = False
        if len(filter_ptss1)==1:
            break
        dists_M = np.ones((len(filter_ptss1), len(filter_ptss1)))
        for i in range(len(filter_ptss1)):
            for j in  range(i+1, len(filter_ptss1)):
                # distance of two clusters
                dist = 1 - polygon_iou(filter_ptss1[i], filter_ptss1[j])
                angle = polygon_angle(filter_ptss1[i], filter_ptss1[j])
                if angle>60:
                    dist=1
                dists_M[i][j] = dist
                dists_M[j][i] = dist

        agg = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                      linkage='average', distance_threshold=(1-iou_thr))
        cluster_idxs = agg.fit_predict(dists_M)
        new_ptss = []
        new_idxs_group = []
        for c_id in range(np.max(cluster_idxs)+1):
            idxs = np.where(cluster_idxs == c_id)[0]
            if idxs.shape[0] >1:
                keep_going = True
            this_ptss = filter_ptss1[idxs[0]]
            this_idxs = idxs_group[idxs[0]]
            for i in range(1, len(idxs)):
                this_ptss = np.concatenate((this_ptss, filter_ptss1[idxs[i]]), axis=0)
                this_idxs = np.concatenate((this_idxs, idxs_group[idxs[i]]), axis=0)
            new_ptss.append(this_ptss)
            new_idxs_group.append(this_idxs)
        filter_ptss1 = new_ptss
        idxs_group = new_idxs_group

    ## remove overlaping
    # dists_M = np.ones((len(filter_ptss1), len(filter_ptss1)))
    # for i in range(len(filter_ptss1)):
    #     for j in  range(i+1, len(filter_ptss1)):
    #         dist = 1 - polygon_iou(filter_ptss1[i], filter_ptss1[j])
    #         dists_M[i][j] = dist
    #         dists_M[j][i] = dist
    # overlap_idxs = np.where(dists_M<1.0)
    # remove_groups=[]
    # for i in range(len(overlap_idxs[0])):
    #     idx_pair = [overlap_idxs[0][i], overlap_idxs[1][i]]
    #
    #     if idx_pair[0] >idx_pair[1]:
    #         continue
    #     if (idx_pair[0] in remove_groups) or (idx_pair[1] in remove_groups):
    #         continue
    #     ## TODO which one to delete
    #     to_remove = np.argmin([len(idxs_group[idx_pair[0]]) ,len(idxs_group[idx_pair[1]])])
    #     if len(idxs_group[idx_pair[to_remove]])<2:
    #         remove_groups.append(idx_pair[to_remove])
    # if len(remove_groups) > 0:
    #     new_idxs_group = []
    #     for i in range(len(idxs_group)):
    #         if i in remove_groups:
    #             continue
    #         new_idxs_group.append(idxs_group[i])
    #     idxs_group = new_idxs_group


    ##TODO yongfeng rm
    ## assign single edge ptss (assign group 2 ptss to clusters)
    # idxs_group2 = np.array(idxs_group2)
    # for i in range(len(filter_ptss2)):
    #     pts = filter_ptss2[i]
    #     id = idxs_group2[i]
    #     merged = False
    #     for g_id in range(len(filter_ptss1)):
    #         g_pts = filter_ptss1[g_id]
    #         all_p_in_group = True
    #         for pt in pts:
    #             if not is_in_2d_polygon(pt, g_pts):
    #                 all_p_in_group = False
    #                 break
    #         if all_p_in_group:
    #             idxs_group[g_id].append(id)
    #             merged = True
    #             break
    ##TODO
    # if merged = False

    # output
    group_pts ={}
    g_id = 0
    ptss =  np.array(ptss)
    idx_map = {}
    for g_ids in idxs_group:
        g_ptss = ptss[g_ids[0]].reshape((1, 4, 3))
        idx_map[g_ids[0]] = g_id
        for k in range(1, len(g_ids)):
            g_ptss = np.concatenate((g_ptss, ptss[g_ids[k]].reshape((1, 4, 3))), axis=0)
            idx_map[g_ids[k]] = g_id
        group_pts[g_id] = g_ptss
        g_id +=1

    return group_pts, idx_map


def filter_overlapping(kpss, idx_map, iou_thr=0.1):
    if len(kpss) == 0:
        return [], idx_map
    iou_M = np.zeros((len(kpss), len(kpss)))
    for i in range(len(kpss)):
        for j in range(i+1, len(kpss)):
            iou = polygon_iou(kpss[i], kpss[j])
            iou_M[i][j] = iou
            iou_M[j][i] = iou
    overlap_idxs = np.where(iou_M > iou_thr)
    remove_groups=[]
    ptss = np.array(kpss)
    for i in range(len(overlap_idxs[0])):
        idx_pair = [overlap_idxs[0][i], overlap_idxs[1][i]]

        if idx_pair[0] >idx_pair[1]:
            continue
        if (idx_pair[0] in remove_groups) or (idx_pair[1] in remove_groups):
            continue
        ## TODO which one to delete
        num0 = np.sum(ptss[idx_pair[0]][:,2])/4.0
        num1 = np.sum(ptss[idx_pair[1]][:,2])/4.0
        to_remove = np.argmin([num0, num1])
        remove_groups.append(idx_pair[to_remove])

    new_kpss = []
    idx_map1={}
    j=0
    for i in range(len(kpss)):
        if i in remove_groups:
            idx_map1[i] = -1
            continue
        idx_map1[i] = j
        new_kpss.append(kpss[i])
        j+=1
    idx_map = file.transfer_maps(idx_map, idx_map1)
    return new_kpss, idx_map

def line_dist(l1, l2):
    return 0.5*min( np.linalg.norm(l1[0] - l2[0]) + np.linalg.norm(l1[1] - l2[1]),
                    np.linalg.norm(l1[0] - l2[1]) + np.linalg.norm(l1[1] - l2[0]))

def box_distance(kps1, kps2):
    kps1 = np.array(kps1)[:,:2]
    kps2 = np.array(kps2)[:,:2]
    p1, p2, p3, p4 = kps1
    pp1, pp2, pp3, pp4 = kps2

    return min(min(line_dist((p1, p4),(pp1, pp4)), line_dist((p2, p3),(pp2, pp3))),
               min(line_dist((p1, p4), (pp2, pp3)), line_dist((p2, p3), (pp1, pp4))))


def filter_too_close_dets(kpss, idx_map):
    if len(kpss) == 0:
        return [], idx_map
    dists_M = np.ones((len(kpss), len(kpss)))*1000
    for i in range(len(kpss)):
        for j in range(i+1, len(kpss)):
            dist = box_distance(kpss[i], kpss[j])
            dists_M[i][j] = dist
            dists_M[j][i] = dist
    overlap_idxs = np.where(dists_M<1.2)
    remove_groups=[]
    ptss = np.array(kpss)
    for i in range(len(overlap_idxs[0])):
        idx_pair = [overlap_idxs[0][i], overlap_idxs[1][i]]

        if idx_pair[0] >idx_pair[1]:
            continue
        if (idx_pair[0] in remove_groups) or (idx_pair[1] in remove_groups):
            continue
        ## TODO which one to delete
        num0 = np.sum(ptss[idx_pair[0]][:,2])/4.0
        num1 = np.sum(ptss[idx_pair[1]][:,2])/4.0
        to_remove = np.argmin([num0, num1])
        remove_groups.append(idx_pair[to_remove])

    new_kpss = []
    idx_map1={}
    j=0
    for i in range(len(kpss)):
        if i in remove_groups:
            idx_map1[i] = -1
            continue
        idx_map1[i] = j
        new_kpss.append(kpss[i])
        j+=1
    idx_map = file.transfer_maps(idx_map, idx_map1)
    return new_kpss, idx_map

def filter_groups_by_area(groups, area_thr=1.0, idx_map={}):
    if groups=={}:
        return {}, idx_map
    filter_result = {}
    idx_map1={}
    for g_id, g_pts in groups.items():
        pts = np.array(g_pts)
        polygon = MultiPoint(pts[:,:,:2].reshape((-1,2))).convex_hull
        area = polygon.area
        if area >area_thr:
            idx_map1[g_id] = g_id
            # id+=1
            filter_result[g_id] = g_pts
        else:
            idx_map1[g_id] = -1
    idx_map = file.transfer_maps(idx_map, idx_map1)
    return filter_result, idx_map

def filter_groups_by_object_num(groups, num_thr=3, idx_map={}):
    filter_result = {}
    idx_map1 = {}
    for g_id, g_pts in groups.items():
        if len(g_pts) >= num_thr:
            idx_map1[g_id] = g_id
            filter_result[g_id] = g_pts
        else:
            idx_map1[g_id] = -1
    idx_map = file.transfer_maps(idx_map, idx_map1)
    return filter_result, idx_map

def merge_groups(groups, idx_map):
    if groups=={}:
        return[], idx_map
    merge_result = []
    idx_map1={}
    i = 0
    for g_id, g_pts in groups.items():
        res = merge_each_group(g_pts)
        # res = merge_each_gt_group(g_pts)
        # res = select_best_det(g_pts)
        if res is not None:
            idx_map1[g_id] = i
            merge_result.append(res)
            i+=1
        else:
            idx_map1[g_id] = -1
    idx_map = file.transfer_maps(idx_map, idx_map1)
    return merge_result, idx_map

def merge_each_group(ptss):
    res = [[0,0,0],[0,0,0],[0,0,0], [0,0,0]]
    nums = [0,0,0,0]
    #TODO YONGFENG Thr
    for pts in ptss:
        for i in range(4):
            if pts[i][2]>keypoint_score_thr0:
                res[i][0] += pts[i][0] * pts[i][2]
                res[i][1] += pts[i][1] * pts[i][2]
                res[i][2] += pts[i][2]
                nums[i] +=1
    count = 0
    for i in range(4):
        if nums[i]>0:
            res[i][0] = res[i][0] / res[i][2]
            res[i][1] = res[i][1] / res[i][2]
            # res[i][2] /= nums[i]
            # res[i][2] = nums[i]
            # count +=1
    # if count<3:
    #     return None
    return res

def select_best_det(ptss):
    if len(ptss) == 0:
        return None
    w=-1
    select_pts = ptss[0]
    for pts in ptss:
        w0=0
        for i in range(4):
            w0 += pts[i][2]
        if w0>w:
            select_pts = pts
            w = w0
    return select_pts


def merge_each_gt_group(ptss):
    res = [[0,0,0],[0,0,0],[0,0,0], [0,0,0]]
    nums = [0,0,0,0]
    #TODO YONGFENG Thr
    for pts in ptss:
        for i in range(4):
            if pts[i][2]>keypoint_score_thr0:
                res[i][0] += pts[i][0]
                res[i][1] += pts[i][1]
                res[i][2] += pts[i][2]
                nums[i] +=1
    count = 0
    for i in range(4):
        if nums[i]>0:
            # res[i][2] /= nums[i]
            res[i][2] = nums[i]
            count +=1
    # if count<3:
    #     return None
    return res

def find_minAreaRects(ptss, idx_map):
    rects = []
    used_ptss = []
    i=0
    j=0
    idx_map1={}
    for pts in ptss:
        # fiter_pts = kps_filter(np.array(pts, dtype=int), 0.1)
        ##TODO yongfeng Thr
        thr = keypoint_score_thr0
        # fiter_pts = kps_filter(np.array(np.multiply(pts, scale), dtype=float), thr)
        fiter_pts = np.array(kps_filter(pts, thr))
        if fiter_pts is None or fiter_pts.shape[0]<3:
            idx_map1[j] = -1
            j+=1
            continue
        scale = 100.0
        rect = cv2.minAreaRect(np.array(np.multiply(fiter_pts, scale), dtype=int))
        cx = rect[0][0]/scale
        cy = rect[0][1]/scale
        w  = rect[1][0]/scale
        h  = rect[1][1]/scale
        rect=((cx, cy),(w, h),rect[2])

        ## for gt refine
        # if j==4:
        #     idx_map1[j] = -1
        #     j+=1
        #     continue
        # if i==2:
        #     rectn =((cx, cy+6), (w, h), rect[2])
        #     rects.append(rectn)
        #     used_ptss.append(pts)
        rect = shrink_rectangle(rect, pts)
        # w, h = rect[1][0], rect[1][1]
        # if w < 0.5 or h < 0.5:
        #     continue
        used_ptss.append(pts)
        rects.append(rect)
        idx_map1[j] = i
        j += 1
        i += 1
    idx_map = file.transfer_maps(idx_map, idx_map1)
    return rects, used_ptss, idx_map


def find_minDistanceRects(ptss, idx_map, l = 2.73, w=car_fix_width, final_merge=False):
    boxes = []
    wheelbases = []
    i=0
    j=0
    idx_map1={}
    wheel_f_b_dist = l
    wheel_l_r_dist = w
    for pts in ptss:
        # fiter_pts = kps_filter(np.array(pts, dtype=int), 0.1)
        ##TODO yongfeng Thr
        thr = keypoint_score_thr0
        # fiter_pts = kps_filter(np.array(np.multiply(pts, scale), dtype=float), thr)
        fiter_pts = np.array(kps_filter(pts, thr))

        # # tmp
        # if j==3 and final_merge==False:
        #     idx_map1[j] = -1
        #     j+=1
        #     continue

        if fiter_pts is None or fiter_pts.shape[0]<3:
            idx_map1[j] = -1
            j+=1
            continue
        (cx, cy) = get_centerpoint(pts)
        rot = 0
        rect = ((cx, cy), (wheel_f_b_dist, wheel_l_r_dist), rot)
        box = cv2.boxPoints(rect).tolist()
        box = np.array([box[1], box[0], box[3], box[2]])
        box = icp(box, np.array(pts)[:,:2])
        # box=np.array(box)
        box = box.reshape(4,2)
        weights = np.array(pts)[:,2].reshape(4,1)
        box = np.concatenate((box, weights), axis=1).tolist()

        wheelbase = get_wheelbase(pts)
        wheelbases.append(wheelbase)
        # box = refine_size_according_to_wheelbase([box], [wheelbase], pct=1.0)[0]

        box=refine_size_use_fix_lw([box], l=l, w=w)[0]

        if final_merge==False:
            box = refine_pose(box, pts)
            # box = refine_size_according_to_wheelbase([box], [wheelbase], pct=1.6391)[0]
            box = refine_size_use_fix_lw([box], l=car_fix_length, w=w)[0]

        ## tmp
        # if j==14 and final_merge==False:
        #     box=np.array(box)
        #     box[:,1] -=1.8
        #     box=box.tolist()

        boxes.append(box)
        idx_map1[j] = i
        j += 1
        i += 1
    idx_map = file.transfer_maps(idx_map, idx_map1)

    return boxes, wheelbases, idx_map

def get_wheelbase(kps):
    pts = np.array(kps)
    if (kps[0][2] + kps[3][2]) > (kps[1][2] + kps[2][2]):
        wheelbase = np.linalg.norm(pts[0][:2] - pts[3][:2])
    else:
        wheelbase = np.linalg.norm(pts[1][:2] - pts[2][:2])
    return wheelbase

def get_centerpoint(pts):
    area = 0.0
    x, y = 0.0, 0.0
    for i in range(len(pts)):
        lat = pts[i][0]  # weidu
        lng = pts[i][1]  # jingdu

        if i == 0:
            lat1 = pts[-1][0]
            lng1 = pts[-1][1]
        else:
            lat1 = pts[i - 1][0]
            lng1 = pts[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0

    x = x / area
    y = y / area
    return (x, y)

def icp(a, b, init_pose=(0,0,0)):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint  to the cloudpoint b.
    '''

    # src = np.array([a.T], copy=True).astype(np.float32)
    # dst = np.array([b.T], copy=True).astype(np.float32)
    src = np.array([a], copy=True).astype(np.float32)
    dst = np.array([b], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    # Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
    #                [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
    #                [0,                    0,                   1          ]])

    # src = cv2.transform(src, Tr[0:2])
    # T = cv2.estimateRigidTransform(src, dst, False)
    # T, inliers = cv2.estimateAffinePartial2D(src, dst, False)
    T, inliers = cv2.estimateAffinePartial2D(src, dst,ransacReprojThreshold=10)
    src = cv2.transform(src, T)
    return src

    # for i in range(no_iterations):
    #     #Find the nearest neighbours between the current source and the
    #     #destination cloudpoint
    #     nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
    #                             warn_on_equidistant=False).fit(dst[0])
    #     distances, indices = nbrs.kneighbors(src[0])
    #
    #     #Compute the transformation between the current source
    #     #and destination cloudpoint
    #     T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
    #     #Transform the previous source and update the
    #     #current source cloudpoint
    #     src = cv2.transform(src, T)
    #     #Save the transformation from the actual source cloudpoint
    #     #to the destination
    #     Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    # return Tr[0:2]

def clip_rects(rects):
    for i in range(len(rects)):
        rect = rects[i]
        w, h = rect[1][0], rect[1][1]
        # w = min(max(w, car_min_width), car_max_height)
        # h = min(max(h, car_min_width), car_max_height)
        ##TODO fix width and height
        if w>h:
            # w = 3.8
            # h = 1.7
            w = car_fix_length
            h = car_fix_width
        else:
            # w = 1.7
            # h = 3.8
            w = car_fix_length
            h = car_fix_width
        rects[i] = (rect[0], (w, h), rect[2])
    return rects

def refine_size_use_fix_lw(ptss, l=car_fix_length, w=car_fix_width):
    ptss=np.array(ptss)
    new_ptss = []
    for pts in ptss:
        l0 = np.linalg.norm(pts[0] - pts[3])
        w0 = np.linalg.norm(pts[0] - pts[1])
        p0 = adjust_vector_len(pts[3][:2], pts[0][:2], l0 + (l - l0)/ 2.0)
        p1 = adjust_vector_len(pts[2][:2], pts[1][:2], l0 + (l - l0) / 2.0)
        p2 = adjust_vector_len(pts[1][:2], pts[2][:2], l0 + (l - l0) / 2.0)
        p3 = adjust_vector_len(pts[0][:2], pts[3][:2], l0 + (l - l0) / 2.0)

        p00 = adjust_vector_len(p1, p0, w0 + (w - w0)/ 2.0)
        p11 = adjust_vector_len(p0, p1, w0 + (w - w0)/ 2.0)
        p22 = adjust_vector_len(p3, p2, w0 + (w - w0)/ 2.0)
        p33 = adjust_vector_len(p2, p3, w0 + (w - w0)/ 2.0)
        new_pts = np.stack([p00, p11, p22, p33], axis=0)
        new_pts = np.concatenate((new_pts, pts[:,2].reshape(4,1)), axis=1)
        new_ptss.append(new_pts.tolist())
    return new_ptss

def refine_size_according_to_wheelbase(ptss, wheelbases, w=car_fix_width, pct=1.6391):
    new_ptss = []
    for i in range(len(ptss)):
        pts = ptss[i]
        l = wheelbases[i] * pct
        new_pts = refine_size_use_fix_lw([pts], l=l, w=w)[0]
        new_ptss.append(new_pts)
    return new_ptss

def adjust_vector_len(p1, p2, len):
    vec = p2-p1
    l0 = np.linalg.norm(p2 - p1)
    if l0==0:
        return None
    e = vec/l0
    p3 = e*len + p1
    return p3

def rects_to_kps(rects):
    kpss = []
    for i in range(len(rects)):
        rect = rects[i]
        box = cv2.boxPoints(rect)
        box = np.flip(box, 0)

        weights = np.array([[1], [1], [1], [1]])
        box_pts = np.concatenate((box, weights), axis=1).tolist()
        kpss.append(box_pts)
    return kpss

def match_box_to_pts(box_ptss, ptss):
    kpss = []
    for i in range(len(box_ptss)):
        pts = ptss[i]
        box_pts = box_ptss[i]
        # box = np.flip(box, 0)
        box_pts, dist = set_box_pts_order(pts, box_pts)

        # weights = np.array([[1],[1],[1],[1]])
        # box_pts = np.concatenate((box, weights), axis=1).tolist()
        kpss.append(box_pts)
    return kpss


def shrink_rectangle(rect, pts):
    dists = np.ones((10, 10),dtype=float) *100000

    for i in range(10):
        for j in range(10):
            ##Todo center shift
            new_rect = (rect[0], (rect[1][0]-i*0.1, rect[1][1]-j*0.1), rect[2])
            box = cv2.boxPoints(new_rect)
            box = np.flip(box, 0)
            box_pts, dist = set_box_pts_order(pts, box)
            # dists[i][j] = dist
            max_diff = distance_diff(box_pts, pts)
            dists[i][j] = max_diff

    i, j = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    print(i, j, dists[i][j], rect[1])
    new_rect = (rect[0], (rect[1][0]-i*0.1, rect[1][1]-j*0.1), rect[2])

    return new_rect

def refine_poses(rect_kpss, ptss):
    new_rect_kpss=[]
    for i in range(len(rect_kpss)):
        rect_kps = refine_pose(rect_kpss[i], ptss[i]).tolist()
        new_rect_kpss.append(rect_kps)
    return new_rect_kpss

def refine_pose(rect_kps, pts):
    num=10
    dists = np.ones((2*num, 2*num), dtype=float)
    scale = 0.1
    rect_kps = np.array(rect_kps)
    for dx in range(2*num):
        for dy in range(2*num):
            new_pts=rect_kps.copy()
            new_pts[:, 0] += (dx-num)*scale
            new_pts[:, 1] += (dy-num)*scale
            w_dist = weighted_distance(new_pts, pts)
            dists[dx][dy] = w_dist

    i, j = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    # print(i, j, dists[i][j], rect_kps[1])
    rect_kps[:, 0] += (i-num)*scale
    rect_kps[:, 1] += (j-num)*scale

    return rect_kps.tolist()

def weighted_distance(box_pts, pts):
    dist = 0
    for i in range(len(box_pts)):
        dist += np.linalg.norm(box_pts[i][:2] - pts[i][:2]) * (pts[i][2])
    return dist

def distance_diff(box_pts, pts):
    dists=[]
    for i in range(len(box_pts)):
        dists.append(np.linalg.norm(box_pts[i][:2] - pts[i][:2]))
    dists=np.array(dists)
    max_diff = np.max(dists)-np.min(dists)
    return max_diff

def set_box_pts_order(pts, box_pts):

    assert len(pts) == len(box_pts)
    pts=np.array(pts)
    box_pts = np.array(box_pts)
    dists = np.zeros(len(pts))
    for shift in range(len(box_pts)):
        shift_box = [box_pts[(j+shift)%len(box_pts)].tolist() for j in range(len(box_pts))]
        dist = ptss_dist(pts, shift_box)
        dists[shift] = dist
    shift = np.argmin(dists)
    shift_box = [box_pts[(j + shift) % len(box_pts)].tolist() for j in range(len(box_pts))]
    shift_box = np.concatenate((np.array(shift_box)[:,:2], pts[:,2].reshape(4,1)), axis=1)
    return shift_box, dists[shift]

def ptss_dist(pts1, pts2):
    dist = 0
    assert len(pts1) == len(pts2)
    for i in range(len(pts1)):
        dist += np.linalg.norm(pts1[i][:2] - pts2[i][:2])
    return dist

def filter_rects(ptss):
    new_ptss = []
    rm = np.zeros(len(ptss)).tolist()
    for i in range(len(ptss)):
        if rm[i] == 1:
            continue
        for j in range(i+1, len(ptss)):
            if rm[j]==1:
                continue
            pts_i = np.array(ptss[i])[:, :2]
            pts_j = np.array(ptss[i])[:, :2]
            iou = polygon_iou(pts_i, pts_j)
            if iou>0:
                rm[j] =1
    for i in range(len(ptss)):
        if rm[i] == 0:
            new_ptss.append(ptss[i])
    return new_ptss


def is_in_2d_polygon(point, vertices):
    px = point[0]
    py = point[1]
    angle_sum = 0

    size = len(vertices)
    if size < 3:
        raise ValueError("len of vertices < 3")
    j = size - 1
    for i in range(0, size):
        sx = vertices[i][0]
        sy = vertices[i][1]
        tx = vertices[j][0]
        ty = vertices[j][1]

        # 通过判断点到通过两点的直线的距离是否为0来判断点是否在边上
        # y = kx + b, -y + kx + b = 0
        k = (sy - ty) / (sx - tx + 0.000000000001)  # 避免除0
        b = sy - k * sx
        dis = fabs(k * px - 1 * py + b) / sqrt(k * k + 1)
        if dis < 0.000001:  # 该点在直线上
            if sx <= px <= tx or tx <= px <= sx:  # 该点在边的两个定点之间，说明顶点在边上
                return True

        # 计算夹角
        angle = atan2(sy - py, sx - px) - atan2(ty - py, tx - px)
        # angle需要在-π到π之内
        if angle >= math.pi:
            angle = angle - math.pi * 2
        elif angle <= -math.pi:
            angle = angle + math.pi * 2

        # 累积
        angle_sum += angle
        j = i

    # 计算夹角和于2*pi之差，若小于一个非常小的数，就认为相等
    return fabs(angle_sum - math.pi * 2) < 0.00000000001

def refine_sharpangle_pts(ptss, angle_thr = 30):
    new_ptss = []
    for pts in ptss:
        angles=[-1 for i in range(num_kps)]
        for i in range(len(pts)):
            pt_i_left = pts[(i - 1)%len(pts)][:2]
            pt_i = pts[i][:2]
            pt_i_right = pts[(i+1)%len(pts)][:2]

            line_a = np.array(pt_i_left)-np.array(pt_i)
            line_b = np.array(pt_i_right)-np.array(pt_i)
            if np.linalg.norm(line_a) * np.linalg.norm(line_b) == 0:
                continue
            # angle = radian_to_degree(get_angle_2pi(line_a, line_b))
            angle = radian_to_degree(get_angle_pi(line_a, line_b))
            angles[i] = angle
            # ##TODO set angle thr
            # thr = 0.15
            # if fabs(angle-90) < angle_thr and pts[i][2]<0.15:
            #     pts[i][2] = 0
        # angles = np.array(angles)
        # max_id = np.argmax(angles)
        # max_angle = np.max(angles)

        min_id = np.argmin(angles)
        min_angle = np.min(angles)
        # angles[min_id] = 360
        # second_min= np.min(angles)
        # if max_angle>180:
        #     pts[max_id] = [pts[(max_id - 1) % 4][0] + pts[(max_id + 1) % 4][0] - pts[(max_id + 2) % 4][0],
        #                    pts[(max_id - 1) % 4][1] + pts[(max_id + 1) % 4][1] - pts[(max_id + 2) % 4][1],
        #                    pts[max_id][2]]
        # if min_angle<45 and angle != -1:
        if min_angle < 45:
            pts[min_id] = [pts[(min_id - 1) % 4][0] + pts[(min_id + 1) % 4][0] - pts[(min_id + 2) % 4][0],
                       pts[(min_id - 1) % 4][1] + pts[(min_id + 1) % 4][1] - pts[(min_id + 2) % 4][1],
                       pts[min_id][2]]
        new_ptss.append(pts)

    return new_ptss


def refine_triangle_pts(kpss):
    new_ptss = []
    for pts in kpss:
        line_len = np.array([-1.0 for i in range(4)])
        for i in range(num_kps):
            pa = np.array(pts[i][:2])
            pb = np.array(pts[(i+1)%num_kps][:2])

            line_len[i] = np.linalg.norm(pa - pb)
        min_dist = np.min(line_len)
        if min_dist<=0:
            continue
        if line_len[0]/line_len[2]<0.1 or  line_len[0]/line_len[2]>10 \
            or line_len[1]/line_len[3] <0.1 or line_len[1]/line_len[3]>10:
            continue
        new_ptss.append(pts)

    return new_ptss

def refine_lowscore_pts(kpss):
    new_kpss=[]
    for kps in kpss:
        thr = keypoint_score_thr
        num_below_thr=0
        id = -1
        for i in range(len(kps)):
            if kps[i][2]<thr:
                id = i
                num_below_thr +=1
            if num_below_thr>1:
                break
        if num_below_thr==1 and id>0:
            kps[id] = [kps[(id-1)%4][0] + kps[(id+1)%4][0] - kps[(id+2)%4][0],
                          kps[(id-1)%4][1] + kps[(id+1)%4][1] - kps[(id+2)%4][1],
                          thr]
            new_kpss.append(kps)
        if num_below_thr==0:
            new_kpss.append(kps)
    return new_kpss

def refine_outside_pts(kpss, mask_3d, idx_map, inside_pct=0.65):
    new_kpss=[]
    idx_map1={}
    id=0
    for i in range(len(kpss)):
        kps=kpss[i]
        pts = np.array(kps)[:,:2]
        box_area_in_mask_pct = get_box_in_mask_pct(pts, mask_3d)
        if box_area_in_mask_pct < inside_pct:
            idx_map1[i] = -1
            continue
        idx_map1[i] = id
        id += 1
        new_kpss.append(kps)

    idx_map = file.transfer_maps(idx_map, idx_map1)

    return new_kpss, idx_map

def get_angle_pi(line_a, line_b):
    ## conter-clockwise
    cosine_angle = np.dot(line_a, line_b) / (np.linalg.norm(line_a) * np.linalg.norm(line_b))
    if math.fabs(cosine_angle - 1.0 ) < 0.000000001:
        return 0
    angle = np.arccos(cosine_angle)
    return angle

def get_angle_2pi(line_a, line_b):
    ## conter-clockwise
    angle = math.atan2(line_b[1], line_b[0]) - math.atan2(line_a[1], line_a[0])
    if (angle < 0):
        angle += 2 * math.pi
    return angle

def radian_to_degree(rad):
    return rad*180/math.pi


def kps_to_parallelogram(kps):
    points = np.array(kps)[:,:2]
    # try:
    #     hull = ConvexHull(points)
    # except:
    #     return None
    new_kps = kps
    # if len(hull.vertices) < 4:
    dist1 = np.linalg.norm(points[0] - points[2])
    dist2 = np.linalg.norm(points[1] - points[3])
    if min(dist1, dist2)< 10 and dist1<dist2:
        center = 0.5*(points[1] + points[3])
        # pt = 0.5*(points[0] + points[2])
        # pt_new = center*2-pt
        if is_clockwise(points[0], points[1], points[3]):
            pt = (center*2-points[0]).tolist()
            pt.append(kps[2][2])
            new_kps = [kps[0], kps[1], pt, kps[3]]
        else:
            pt = (center*2-points[2]).tolist()
            pt.append(kps[0][2])
            new_kps = [pt, kps[1], kps[2], kps[3]]
    elif min(dist1, dist2)< 10:
        center = 0.5*(points[0] + points[2])
        if is_clockwise(points[0], points[1], points[2]):
            pt=(center*2-points[3]).tolist()
            pt.append(kps[1][2])
            new_kps = [kps[0], pt, kps[2], kps[3]]
        else:
            pt = (center * 2 - points[1]).tolist()
            pt.append(kps[3][2])
            new_kps = [kps[0], kps[1], kps[2], pt]
    # else:
    #     dist=np.zeros(4)
    #     verts = hull.vertices
    #     for i in range(4):
    #         dist[i] = np.linalg.norm(points[verts[i]] - points[verts[(i+1)%4]])
    #     min_dist = np.argmin()

    return new_kps

def kps_in_bbox(kps, bbox):
    x1, y1, x2, y2 = bbox
    h = y2-y1
    for i in range(len(kps)):
        # kps[i][0] = min(max(x1, kps[i][0]), x2)
        # kps[i][1] = min(max(int(y1 + h/2), kps[i][1]), y2)
        kps[i][1] = max(int(y1 + h / 2), kps[i][1])
    return kps

def is_clockwise(pt1, pt2, pt3):
    if ((pt2[0]- pt1[0])*(pt3[1]*pt1[1]) - (pt3[0]- pt1[0])*(pt2[1]*pt1[1])) <0:
        return True
    else:
        return False

def distance(pts1, pts2):
    minPt1, minPt2, minDist = find_nearest_pts(pts1, pts2)
    return minDist

def find_nearest_pts(pt1, pt2):
    # The input are two lists of points defining the two polygons of interest
    # Shorten some function names that will be repeated
    nr = np.roll
    nt = np.tile

    # Give lists as array
    arr1 = np.array(pt1)
    arr2 = np.array(pt2)
    A0 = arr1[:-1, 0]  # array of X for pt1, last point is the same of the first
    A1 = arr1[:-1, 1]  # array of Y for pt2

    # Roll arrays to serve as the second point to form lines
    # each [[A0,A1],[B0,B1]] form a line
    B0 = nr(A0, 1)
    B1 = nr(A1, 1)

    # Do the same for points of the second polygon
    C0 = arr2[:-1, 0]
    C1 = arr2[:-1, 1]
    D0 = nr(C0, 1)
    D1 = nr(C1, 1)
    size1 = len(A0)
    size2 = len(C0)

    # Get tiled and repeated arrays to get all possible pairs of points
    A0 = nt(A0, size2)
    A1 = nt(A1, size2)
    B0 = nt(B0, size2)
    B1 = nt(B1, size2)
    C0 = C0.repeat(size1)
    C1 = C1.repeat(size1)
    D0 = D0.repeat(size1)
    D1 = D1.repeat(size1)

    '''
    Throw the arrays into nearestPoint function
    Every pair of nearest points will be either:
     - Points that define the polygons.
     - A point that define a polygon and a projected
     point into a line that define the other polygon.
    '''
    result1 = nearestPoint(A0, A1, B0, B1, C0, C1)
    result2 = nearestPoint(C0, C1, D0, D1, A0, A1)

    # Return the result that is nearest
    if result1[2] < result2[2]:
        return result1
    else:
        return result2


def nearestPoint(A0, A1, B0, B1, C0, C1):
    '''
    This function is adapted from another non-vectorized implementation:
    <http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment>

    The inputs are numpy arrays:
     - A0: array of X values for point A
     - A1: array of Y values for point A
     - B0: array of X values for point B
     - B1: array of Y values for point B
     - C0: array of X values for point C
     - C1: array of Y values for point C
    '''
    size = A0.size
    AB0 = B0 - A0
    AB1 = B1 - A1
    AB_squared = (AB0 * AB0 + AB1 * AB1)
    CA0 = C0 - A0
    CA1 = C1 - A1
    t = (CA0 * AB0 + CA1 * AB1) / AB_squared
    result = np.array([[np.nan, np.nan]]).repeat(size, 0)
    test = t < 0
    result[test] = zip(A0[test], A1[test])
    test2 = t > 1
    result[test2] = zip(B0[test2], B1[test2])
    test3 = np.logical_not(test | test2)
    if sum(test3) != 0:
        result[test3] = zip((A0[test3] + t[test3] * AB0[test3]), (A1[test3] + t[test3] * AB1[test3]))
    dists = ((result[:, 0] - C0) ** 2 + (result[:, 1] - C1) ** 2)
    minDist = min(dists) ** 0.5
    minIndex = dists.argmin()
    minPt1 = (C0[minIndex], C1[minIndex])
    minPt2 = result[minIndex]
    return (minPt1, (minPt2[0], minPt2[1]), minDist)
