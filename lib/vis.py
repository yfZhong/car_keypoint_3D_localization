#!/usr/bin/env python
"""
Functions for visualiation.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import cv2
import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt

keypoint_score_thr = 0.0

def save_img(img, path):
    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    cv2.imwrite(path, img)

def vis_car_keypoints(img, kps, kp_thresh=0.0, alpha=0.7, idx=-1):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    #dataset_keypoints = PersonKeypoints.NAMES
    #kp_lines = PersonKeypoints.CONNECTIONS
    # dataset_keypoints = ['front', 'rear']
    # kp_lines =[[0,1], [1,3], [3,2], [2,0]]
    kp_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + len(kps))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    for i in range(len(kps)):
        pi = tuple((int(kps[i][0]), int(kps[i][1])))
        vis = float(kps[i][2])
        # vis = 3
        # if len(kps[i])==3:
        #     vis = float(kps[i][2])
        # print(vis)
        # if vis > kp_thresh:
        #     cv2.circle(
        #         kp_mask, pi,
        #         radius=6, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        if vis == float(1):# visible
            cv2.circle(
                kp_mask, pi,
                radius=4, color=(0,0,0), thickness=4, lineType=cv2.LINE_AA)
        elif vis == float(2):
            cv2.circle(
                kp_mask, pi,
                radius=4, color=(255,255,255), thickness=4, lineType=cv2.LINE_AA)
    # # Draw the keypoint lines.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = tuple((int(kps[i1][0]), int(kps[i1][1])))
        p2 = tuple((int(kps[i2][0]), int(kps[i2][1])))
        vis1 = float(kps[i1][2])
        vis2 = float(kps[i2][2])
        # vis1, vis2 = 0, 0
        # if len(kps[i1])==3:
        #     vis1 = float(kps[i1][2])
        # if len(kps[i2])==3:
        #     vis2 = float(kps[i2][2])
        if vis1 <= kp_thresh or vis2 <= kp_thresh:
            continue
        cv2.line(kp_mask, p1, p2, color=colors[l+len(kps)], thickness=7, lineType=cv2.LINE_AA)

    if idx !=-1:
        center = (0.5*np.add(kps[0], kps[2]))[:2]
        loc = (int(center[0])-50, int(center[1])+50)
        if idx>=10:
            loc = (int(center[0]) - 100, int(center[1]) + 50)

        cv2.putText(kp_mask, str(int(idx)), loc, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,144,30), 15)

    res = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    # Blend the keypoints.
    return res

def vis_polygon(img, ptss, color=(200, 200, 100)):
    for i in range(len(ptss)):
        p1 =tuple((int(ptss[i][0]), int(ptss[i][1])))
        p2 =tuple((int(ptss[(i+1)%len(ptss)][0]), int(ptss[(i+1)%len(ptss)][1])))
        cv2.line(img, p1, p2, color=color, thickness=7, lineType=cv2.LINE_AA)
    return img

def vis_bbox_from3d(img, kps, kp_thresh=0.0, alpha=0.7, idx=-1):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    #dataset_keypoints = PersonKeypoints.NAMES
    #kp_lines = PersonKeypoints.CONNECTIONS
    # dataset_keypoints = ['front', 'rear']
    # kp_lines =[[0,1], [1,3], [3,2], [2,0]]
    kp_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + len(kps))]
    colors = [(c[2] * 150, c[1] * 150, c[0] * 150) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    for i in range(len(kps)):
        pi = tuple((int(kps[i][0]), int(kps[i][1])))
        vis = float(kps[i][2])
        # vis = 3
        # if len(kps[i])==3:
        #     vis = float(kps[i][2])
        # print(vis)
        if vis > kp_thresh:
            cv2.circle(
                kp_mask, pi,
                radius=6, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        # elif vis == int(1):
        #     cv2.circle(
        #         kp_mask, pi,
        #         radius=6, color=colors[i], thickness=3, lineType=cv2.LINE_AA)
    # # Draw the keypoint lines.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = tuple((int(kps[i1][0]), int(kps[i1][1])))
        p2 = tuple((int(kps[i2][0]), int(kps[i2][1])))
        vis1 = float(kps[i1][2])
        vis2 = float(kps[i2][2])
        # vis1, vis2 = 0, 0
        # if len(kps[i1])==3:
        #     vis1 = float(kps[i1][2])
        # if len(kps[i2])==3:
        #     vis2 = float(kps[i2][2])
        if vis1 <= kp_thresh or vis2 <= kp_thresh:
            continue
        cv2.line(kp_mask, p1, p2, color=colors[l+len(kps)], thickness=7, lineType=cv2.LINE_AA)

    if idx !=-1:
        center = (0.5*np.add(kps[0], kps[2]))[:2]
        loc = (int(center[0])-50, int(center[1])+50)
        if idx>=10:
            loc = (int(center[0]) - 100, int(center[1]) + 50)

        cv2.putText(kp_mask, str(int(idx)), loc, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,144,30), 15)

    res = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    # Blend the keypoints.
    return res

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = np.array(labels)[:, None] * np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = (colors*10 % 255).astype("uint8")
    return colors

def vis_box(image, preds):
    image =image.copy()
    boxes = preds.bbox
    labels = preds.get_field("labels").tolist()
    scores = preds.get_field("scores").to(torch.float32).tolist()
    colors = compute_colors_for_labels(labels).tolist()
    vis_id=False
    if preds.has_field('idxs'):
        ids=preds.get_field("idxs").to(torch.int).tolist()
        vis_id=True
    i=0
    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )
        area = (box[2] - box[0]) * (box[3] - box[1])
        height = box[3] - box[1]
        cv2.putText(image, str(int(height)), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.putText(image, str(float(scores[i]))[:5], (box[0] + 10, box[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        if vis_id:
            cv2.putText(image, str(ids[i]), ((box[2] + box[0])/2-20, (box[3] + box[1])/2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 200, 100), 10)
        i+=1
    return image

def vis_group(group_kpss, path):
    scale=50
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(group_kpss.keys()) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    box_mask = np.zeros((80 * scale, 80 * scale, 3), np.uint8)
    g_id = 0
    for id, kpss in group_kpss.items():
        kpss=np.array(kpss)
        shift = np.ones(kpss.shape) * 40
        shift[:, :, 2] = 0
        kpss = (kpss + shift) * scale
        for b in range(len(kpss)):
            kps = kpss[b]
            for l in range(len(kps)):
                p1 = tuple((int(kps[l][0]), int(kps[l][1])))
                p2 = tuple((int(kps[(l+1)%len(kps)][0]), int(kps[(l+1)%len(kps)][1])))
                # vis1 = float(kpss[i][2])
                # vis2 = float(kpss[i][2])
                # if vis1 <= kp_thresh or vis2 <= kp_thresh:
                #     continue

                cv2.line(box_mask, p1, p2, color=colors[int(g_id)], thickness=7, lineType=cv2.LINE_AA)
        cv2.putText(box_mask, str(int(id)), (int(kpss[0][0][0]) - 0, int(kpss[0][0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 5, 255, 5)
        g_id+=1
    box_mask = cv2.flip(box_mask, 0)
    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    cv2.imwrite(path, box_mask)

def vis_masks(group_kpss, path):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(group_kpss.keys()) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    box_mask = np.zeros((80 * 20, 80 * 20, 3), np.uint8)
    g_id = 0
    for id, kps in group_kpss.items():
        kps=np.array(kps)
        shift = np.ones(kps.shape) * 40
        shift[:, 2] = 0
        kps = (kps + shift) * 20

        for l in range(len(kps)):
            p1 = tuple((int(kps[l][0]), int(kps[l][1])))
            p2 = tuple((int(kps[(l+1)%len(kps)][0]), int(kps[(l+1)%len(kps)][1])))
            # vis1 = float(kpss[i][2])
            # vis2 = float(kpss[i][2])
            # if vis1 <= kp_thresh or vis2 <= kp_thresh:
            #     continue
            cv2.line(box_mask, p1, p2, color=colors[int(g_id)], thickness=5, lineType=cv2.LINE_AA)
        g_id+=1
    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    cv2.imwrite(path, box_mask)

def vis_mask(kps, path, color):

    mask = np.zeros((80 * 20, 80 * 20, 3), np.uint8)
    kps=np.array(kps)
    shift = np.ones(kps.shape) * 40
    shift[:, 2] = 0
    kps = (kps + shift) * 20
    for l in range(len(kps)):
        p1 = tuple((int(kps[l][0]), int(kps[l][1])))
        p2 = tuple((int(kps[(l+1)%len(kps)][0]), int(kps[(l+1)%len(kps)][1])))
        cv2.line(mask, p1, p2, color=color, thickness=5, lineType=cv2.LINE_AA)
    mask = cv2.flip(mask, 0)
    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    cv2.imwrite(path, mask)

def vis_kps_in_img(img, preds, path):
    img = vis_box(img, preds)
    keypointss = preds.get_field("keypoints")
    kpss = keypointss.keypoints.to(torch.float32).tolist()
    # kpss = preds.get_field("keypoints").to(torch.float32).tolist()

    # TODO yongfeng
    # for kps in kpss:
    #     img = vis_car_keypoints(img, kps, 0)
    # if preds.has_field('kps_from_3d'):
    #     kps_from_3d = preds.get_field('kps_from_3d').to(torch.int64).tolist()
    #     for kps in kps_from_3d:
    #         img = vis_car_keypoints(img, kps, 0)

    if preds.has_field('bbox_from_3d'):
        bboxes_from_3d = preds.get_field('bbox_from_3d').to(torch.int64).tolist()
        for kps in bboxes_from_3d:
            img = vis_bbox_from3d(img, np.concatenate([np.array(kps), np.ones((4,1))], axis=1).tolist(), 0)

    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    cv2.imwrite(path, img)

def vis_kps_on_blank_img(kps_3dflatsss, path, mask=[]):
    if len(kps_3dflatsss)==0:
        return
    scale = 50
    blank_img = np.zeros((80 * scale, 80 * scale, 3), np.uint8)
    blank_img[:, :] = (100, 100, 100)
    kps_3dflatsss = np.array(kps_3dflatsss)
    shift = np.ones(kps_3dflatsss.shape) * 40
    shift[:, :, 2] = 0
    kps_3dflatsss = (kps_3dflatsss + shift) * scale
    kps_3dflatsss = list(kps_3dflatsss)
    for kps_3dflats in kps_3dflatsss:
        blank_img = vis_car_keypoints(blank_img, kps_3dflats, keypoint_score_thr * 20)
    if (len(mask))>0:
        mask=np.array(mask)
        shift = np.ones(mask.shape) * 40
        shift[:, 2] = 0
        blank_img = vis_polygon(blank_img, (np.array(mask)+shift)*scale)
    blank_img = cv2.flip(blank_img, 0)

    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)

    cv2.imwrite(path, blank_img)

def vis_2dgroundimg(img, ptss2d, path, ids=[]):

    for i in range(len(ptss2d)):
        pts2d=ptss2d[i]
        idx = i+1
        if len(ids)>0:
            idx=ids[i]
        img = vis_car_keypoints(img, pts2d, kp_thresh=-1, alpha=1, idx=idx)
    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    cv2.imwrite(path, img)

def vis_pred_ress(pred_ress, img_root, save_root):
    for res in pred_ress:
        preds = res['preds']
        img_path=osp.join(img_root,res['img_path'])
        vis_path = osp.join(save_root, res['vis_path'])
        img = cv2.imread(img_path, 1)
        vis_kps_in_img(img, preds, vis_path)

        ids = preds.get_field("idxs").to(torch.int).tolist()
        for id in ids:
            if id<=0:
                continue
            lid_pkg = osp.join("/".join(vis_path.split("/")[:-1]), str(id))
            if (not osp.exists(lid_pkg)):
                mkdir = "mkdir -p \"" + lid_pkg + "\""
                os.system(mkdir)
            cmd = 'cp ' + vis_path + " " + lid_pkg
            os.system(cmd)

def vis_pred_crops(pred_ress, img_root, pose_root):
    dic_id_count = {}
    for res in pred_ress:
        img_path = osp.join(img_root, res['img_path'])
        img = cv2.imread(img_path, 1)
        preds = res['preds']
        boxes = preds.bbox.to(torch.int64).tolist()
        ids = preds.get_field("idxs").to(torch.int).tolist()
        i=0
        crop_img_root=osp.join(pose_root, "crop_images")
        if (not osp.exists(crop_img_root)):
            mkdir = "mkdir -p \"" + crop_img_root + "\""
            os.system(mkdir)

        for i in range(len(boxes)):
            id = ids[i]
            if id <= 0:
                continue
            if id not in dic_id_count.keys():
                dic_id_count[id]=0
            vis_path = osp.join(crop_img_root, str(id)+"_"+str(dic_id_count[id]) + ".jpg")
            dic_id_count[id] += 1

            box = boxes[i]
            # box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            box = [round(box[0]), round(box[1]), round(box[2]), round(box[3])]
            box_crop = img[box[1]:box[3], box[0]:box[2], :]

            save_dir = osp.dirname(vis_path)
            if (not osp.exists(save_dir)):
                mkdir = "mkdir -p \"" + save_dir + "\""
                os.system(mkdir)
            cv2.imwrite(vis_path, box_crop)