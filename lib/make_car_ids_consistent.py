#!/usr/bin/env python
"""
Functions for keeping car id consistant with history result.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import cv2
import os
import os.path as osp
import _init_paths
import proj
import argparse
import boxlist_process
import polygon
import pred_keypoints
import file
import vis
import glob
import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy as np
import calculate_pose_pr
from scipy.optimize import linear_sum_assignment
import json
from init_cameras import load_cameras

def parse_args():
    parser = argparse.ArgumentParser(description='process json.')
    parser.add_argument('--previous_pose_root', type=str, help='previous_pose_root')
    parser.add_argument('--current_pose_root', type=str, help='current_pose_root')
    parser.add_argument('--new_pose_root', type=str, help='current_pose_root')
    parser.add_argument('--program_time_list', type=str, help='input list')
    # parser.add_argument('--output_json', type=str, help='output_json')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    previous_pose = calculate_pose_pr.fomat_det(args.previous_pose_root, args.program_time_list)
    current_pose = calculate_pose_pr.fomat_det(args.current_pose_root, args.program_time_list)
    id_maps = match(previous_pose, current_pose)
    program_time_list = file.load_lines(args.program_time_list)
    for info in program_time_list:
        print(info)
        customer, city, store, day, t = info.split('\t')
        path = osp.join(customer, city, store)
        camera_info_root = osp.join("./CameraInfos", path)
        if store == "qybc" and int(day) < 20190929:
            camera_info_root += "_before_20190929"
        if store == "hddb" and int(day) < 20191103:
            camera_info_root += "_before_20191103"
        if store == "bydz" and int(day) < 20191209:
            camera_info_root += "_before_20191209"
        elif store == "bydz" and int(day) < 20191222:
            camera_info_root += "_before_20191222"

        # load camera config
        ch_names = os.listdir(osp.join(camera_info_root, "regular"))
        camera_cfg = load_cameras(
            db='nanling', camerainfo_root=camera_info_root, ch_names=ch_names)

        # floorimg_path = osp.join(camera_info_root, 'floorinfos', 'floor.jpg')
        # floorimg = cv2.imread(floorimg_path)

        # previous_pose_root = osp.join(args.previous_pose_root, customer, city, store, 'car', 'pose', day)
        current_pose_root = osp.join(args.current_pose_root, customer, city, store, 'car', 'pose', day)
        new_pose_root = osp.join(args.new_pose_root, customer, city, store, 'car', 'pose', day)

        if (not osp.exists(new_pose_root)):
            mkdir = "mkdir -p \"" + new_pose_root + "\""
            os.system(mkdir)

        pose_path0 = osp.join(customer, city, store, "car", "pose", day, "pose.json")
        id_map = id_maps[pose_path0]

        cp_vis = "cp -r "+current_pose_root+"/vis " + new_pose_root
        os.system(cp_vis)

        # update pose.json
        poses_res = json.load(open(osp.join(current_pose_root, 'pose.json'), 'r'))
        ids = []
        poses = []
        for pose in poses_res:
            id = pose['id']
            ids.append(id_map[id])
            poses.append(pose['cords'])

        file.save_kpss(poses, osp.join(new_pose_root, 'pose.json'), ids)
        weights = np.ones((len(poses),4,1))
        poses = np.concatenate((np.array(poses), weights), axis=2).tolist()
        # kps_2dgroundss = proj.map_3dkpss_to_2dgroundimg(camera_cfg, poses)
        vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), poses, osp.join(new_pose_root, "pose.jpg"), ids)
        img_paths = glob.glob(osp.join(current_pose_root, "crop_images", "*.jpg"))
        save_dir = osp.join(new_pose_root, "crop_images")
        if (not osp.exists(save_dir)):
            mkdir = "mkdir -p \"" + save_dir + "\""
            os.system(mkdir)

        for img_path in img_paths:
            img_name = img_path.split("/")[-1]
            lid, cid_name = img_name.split("_")
            if int(lid) not in id_map.keys():
                continue
            lid = id_map[int(lid)]
            img_name = "_".join([str(lid), cid_name])
            save_path = osp.join(new_pose_root, "crop_images", img_name)
            cp = 'cp ' + img_path +' ' + save_path
            os.system(cp)

        # pred_ress = json.load(open(osp.join(current_pose_root, 'pred_result.json'), 'r'))
        #
        # print(pose_path0)
        # for res in pred_ress:
        #     new_ids=[]
        #     ids = res['ids']
        #     for id in ids:
        #         if id==-1 or (id not in id_map.keys()):
        #             new_ids.append(-1)
        #         else:
        #             new_ids.append(id_map[id])
        #     res['ids'] = new_ids
        #
        # with open(osp.join(new_pose_root, 'pred_result.json'), 'w') as outfile:
        #     json.dump(pred_ress, outfile, indent=4)


def match(previous_pose, current_pose):
    id_maps={}
    for img_name in previous_pose:
        pre_poses = previous_pose[img_name]['dets']
        pre_ids = previous_pose[img_name]['ids']
        if img_name in current_pose:
            cur_poses = current_pose[img_name]['dets']
            cur_ids = current_pose[img_name]['ids']
            # print(img_name)
            if pre_poses ==[]:
                # i = 1
                id_map={}
                for id in cur_ids:
                    id_map[id] = id
                    # i+=1
                id_maps[img_name] = id_map
                continue
            iou_matrix = calculate_pose_pr.single_iou_matrix(cur_poses, pre_poses)
            row_idxs, col_idxs = linear_sum_assignment(-iou_matrix)
            real_cur_idxs = [cur_ids[row_idxs[i]] for i in range(len(row_idxs))]
            real_pre_idxs = [pre_ids[col_idxs[i]] for i in range(len(col_idxs))]

            # for i in range(len(real_cur_idxs)):
            #     if real_pre_idxs[i]==-1:
            #         real_pre_idxs[i]=np.max(np.array(real_pre_idxs))+1
            id_map = {real_cur_idxs[i]:real_pre_idxs[i] for i in range(len(real_cur_idxs))}
            max_id = np.max(np.array(real_pre_idxs)).tolist()
            for id in cur_ids:
                if id not in id_map.keys():
                    id_map[id] = max_id+1
                    max_id+=1

            id_maps[img_name] = id_map

    return id_maps


if __name__ == "__main__":
    main()
