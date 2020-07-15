# /usr/bin/env python
"""
Functions for generating the car pose with video input.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import time
import os
import os.path as osp
import numpy as np
import math
import _init_paths
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from init_cameras import load_cameras
from proj import project_2d_to_3dflat, project_3dflat_to_2dground
from args import parse_args
from boxlist_process import filter_preds, load_img_gts_boxlist, load_lines, filter_preds_by_mask
from vis import vis_kps_on_blank_img, vis_2dgroundimg,vis_kps_in_img
from polygon import polygons_cluster,polygons_cluster2,merge_groups, find_minAreaRects, filter_rects, \
    filter_pts_by_angle, filter_groups_by_area, clip_rects, rects_to_kps,filter_groups_by_object_num
import datetime
import json

size=(2560, 1440)

def main():
    args = parse_args()

    # load camera config
    camera_cfg = load_cameras(
        db='nanling', camerainfo_root=args.camera_info_root, ch_names=args.ch_names)

    # load model config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    videos = {}
    time = args.time
    if args.use_current_time=="True":
        d = datetime.datetime.today()
        day="{}{}{}".format(d.year, d.month, d.day)
        time = "{}180000".format(day)


    for ch_name in args.ch_names:
        video_name = "{}_{}.mp4.cut.mp4".format(ch_name, time)
        hdfs_video_path = osp.join(args.hdfs_video_root, time[:8], video_name)
        path = osp.join(args.video_root, time[:8], video_name)

        if not osp.exists(path):
            pkg = osp.dirname(path)
            if (not osp.exists(pkg)):
                mkdir = "mkdir -p \"" + pkg + "\""
                os.system(mkdir)
            get_data = "hdfs dfs -get {} {}".format(hdfs_video_path, path)
            os.system(get_data)

        if osp.exists(path):
            print(path)
            videos[ch_name] = cv2.VideoCapture(path)
            videos[ch_name].set(cv2.CAP_PROP_FPS, 1)
        else:
            print(path + " not exits")
    id = 0
    global_kpsss = []
    frame_step = 1000

    to_continue = True
    while to_continue:
        print(id)
        kps_3dflatsss = []
        box_scoress = []
        kps_2dgroundsss = []
        for ch_name, video in videos.items():
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_step * id)
            to_continue, img = video.read()
            # ret_val, img = video.read()
            if img is None:
                to_continue = False
                break
            composite, preds = coco_demo.run_on_opencv_image(img)
            box_height_thr = 300
            box_width_thr = 300
            area_thr = 1000
            img_name = "{}_{}.jpg".format(ch_name, id)
            vis_kps_in_img(img, preds, args.save_root + "/img/" + img_name)
            preds = filter_preds(preds, box_height_thr, box_width_thr, area_thr)
            if preds == None:
                continue
            preds = filter_preds_by_mask(preds, camera_cfg.mask[ch_name])
            if preds == None:
                continue

            vis_kps_in_img(img, preds, args.save_root + "/img_filter/" + img_name)
            kps_3dflatss, box_scores = map_kps_to_3d(preds, camera_cfg, ch_name)

            kps_2dgroundss = map_3dkpss_to_2dgroundimg(camera_cfg, kps_3dflatss)

            img_name = "{}_{}_{}.jpg".format(time, ch_name, id)



            if args.vis_3d_single=="True":
                # vis_kps_on_blank_img(kps_3dflatss, args.save_root +"/blankimg/"+ img_name)
                vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, args.save_root +"/groundimg/"+ img_name)

            kps_3dflatsss += kps_3dflatss
            box_scoress += box_scores
            kps_2dgroundsss += kps_2dgroundss
        if kps_3dflatsss == []:
           break
        if args.vis_3d_merged=="True":
            # vis_kps_on_blank_img(np.array(kps_3dflatsss)[:,:4].tolist(), args.save_root + "/blankimg/" + time + ".jpg")
            vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundsss, args.save_root + "/groundimg/" + time +"_"+ str(id)+ ".jpg")

        vis_kps_on_blank_img(kps_3dflatsss, args.save_root + "/merged/" + time+ "_"+ str(id)+ ".jpg")
        # groups = polygons_cluster(kps_3dflatsss, box_scoress, iou_thr=0.0)
        groups = polygons_cluster2(kps_3dflatsss, iou_thr=0.0)
        groups = filter_groups_by_area(groups, area_thr=0.2)

        merge_result = merge_groups(groups)
        vis_kps_on_blank_img(merge_result, args.save_root + "/merge_result/" + time + "_"+ str(id)+".jpg")

        rects, ptss = find_minAreaRects(merge_result)

        rects = clip_rects(rects)
        kpss = rects_to_kps(rects, ptss)

        vis_kps_on_blank_img(kpss, args.save_root + "/result/" + time +"_" + str(id)+ ".jpg")

        # res2 = filter_rects(ress)
        # vis_kps_on_blank_img(res2, args.save_root + "/result2/" + time + ".jpg")
        kps_2dgroundss = map_3dkpss_to_2dgroundimg(camera_cfg, kpss)
        vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, args.save_root + "/result_groundimg/" + time+"_" + str(id) + ".jpg")
        # break
        id +=1
        global_kpsss.extend(kpss)

    global_groups = polygons_cluster2(global_kpsss, iou_thr=0.6)
    global_groups = filter_groups_by_object_num(global_groups, id * 0.3)
    merged = merge_groups(global_groups)

    vis_kps_on_blank_img(merged, args.save_root + "/merged_group/" + time + "_" + str(id) + ".jpg")
    rects, ptss = find_minAreaRects(merged)
    rects = clip_rects(rects)
    kpss = rects_to_kps(rects, ptss)
    save_kpss(kpss, args.output_json)

    kps_2dgroundss = map_3dkpss_to_2dgroundimg(camera_cfg, kpss)
    vis_kps_on_blank_img(kpss, args.save_root + "/final_result_groundimg/" + time + "_blank_" + str(id) + ".jpg")
    vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, args.save_root + "/final_result_groundimg/" + time+"_" + str(id) + ".jpg")
    save_kpss(kps_2dgroundss, "data/position2.json")
    for ch_name, video in videos.items():
        videos[ch_name].release()

def save_kpss(kpss, path):
    output=[]
    for kps in kpss:
        data = {}
        data['cords'] = np.array(kps)[:,:2].tolist()
        output.append(data)
    with open(path, 'w') as outfile:
        json.dump(output,outfile, indent=4)


def map_kps_to_3d(preds, camera_cfg, ch_name):
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

        box_scores.append(polygon_score)
        kps_3dflatss.append(kps_3dflats)
    return kps_3dflatss, box_scores


def map_3dkpss_to_2dgroundimg(camera_cfg, ptss3d):
    ptss2d = []
    for pts3d in ptss3d:
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

def gen_channel_img_dic(img_list):
    ## read to dic
    dic={}
    for line in img_list:
        'NANLING_guangzhou_qybc_ch01012_20190915180000.mp4.cut.mp4_002336.jpg'
        ch = line.split("_")[3]
        date = line.split("_")[4][:8]
        time = line.split("_")[4][8:8+4]
        key = date + "_" + time
        if key not in dic.keys():
            dic[key] = {}
        # if ch in dic[key].keys():
        #     print("repeated ! {}, {}".format(key, ch))
        dic[key][ch] = line
    return dic

if __name__ == "__main__":
    main()
