#!/usr/bin/env python
"""
Functions of the main processiong: loading images and calculating car poses in 3D ground
History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-11
"""

import cv2
import os
import os.path as osp
import _init_paths
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from init_cameras import load_cameras
import proj
from args import parse_args
import boxlist_process
import polygon
import pred_keypoints
import file
import vis
import glob
import torch

def init_model(config_file, conf_thr, min_image_size, opts):
    # load model config from file and command-line arguments
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=conf_thr,
        min_image_size=min_image_size
    )
    return coco_demo

def main():
    args = parse_args()
    program_time_list = file.load_lines(args.program_time_list)

    # load model config from file and command-line arguments
    box_model = init_model(args.config_file_box, args.confidence_threshold, args.min_image_size, args.opts)
    kp_model = init_model(args.config_file_kp, args.confidence_threshold, args.min_image_size_kp, args.opts)

    for info in program_time_list:
        print(info)
        customer, city, store, day, t = info.split('\t')
        path = osp.join(customer, city, store)
        camera_info_root = osp.join("./CameraInfos", path)

        # load camera configl
        ch_names = os.listdir(osp.join(camera_info_root, "regular"))
        camera_cfg = load_cameras(
            db='nanling', camerainfo_root=camera_info_root, ch_names=ch_names)

        if args.vis_gts=="True" and args.gt_annos is not None:
            dic_img_gts = boxlist_process.load_img_gts_boxlist(args.gt_annos)

        img_path_root = osp.join(args.img_root, customer, city, store, 'car', 'images', day)
        pose_root = osp.join(args.save_root, customer, city, store, 'car', 'pose', day)
        vis_root = osp.join(pose_root, 'vis')
        post_process_root = osp.join(vis_root, 'post_process')
        global_kpsss = []
        global_kpsss_ld = []
        t=t[:6]
        num_frames = 30
        final_save_name = t + ".jpg"
        used_frame_num = 0
        pred_resss = []

        for i in range(num_frames):
            res={}
            print("frame_id: {}".format(i))
            t1 = str(int(t)+(i*100))
            kps_3dflatsss = []
            kps_3dflatsss_lid = []
            img_name_tmp = day + t1[:4] + "*.jpg"
            # img_name_tmp = day + "*.jpg"
            save_name = t1 +"_"+str(i)+ ".jpg"
            pred_ress=[]
            for ch_name in ch_names:
                img_path = (osp.join(img_path_root, ch_name, img_name_tmp))
                img_path = glob.glob(img_path)
                if len(img_path) == 0:
                    continue
                img_path = img_path[0]
                img_name = os.path.basename(img_path)
                if not osp.exists(img_path):
                    continue
                img = cv2.imread(img_path, 1)
                if img is None:
                    continue
                if args.vis_gts=="True":
                    img_key = osp.join(customer, city, store, 'car', 'images', day, ch_name, img_name)
                    if img_key not in dic_img_gts.keys():
                        continue
                    gts = dic_img_gts[img_key]
                    preds = gts
                else:
                    #do img prediction
                    cfg.merge_from_file(args.config_file_box)
                    composite, preds = box_model.run_on_opencv_image(img)
                    cfg.merge_from_file(args.config_file_kp)
                    preds = pred_keypoints.get_boxes_kps_pad(img, preds, kp_model, score_thr=box_score_thr)

                preds = boxlist_process.filter_preds(preds, box_height_thr, box_width_thr, area_thr)
                preds = proj.add_3d_keypoints(preds, camera_cfg, ch_name, args.vis_gts)
                if preds == None:
                    continue

                projected_kpss = preds.get_field('keypoints_3d').to(torch.float).tolist()
                idxs = preds.get_field('idxs').to(torch.int).tolist()
                idx_map = {j:j for j in idxs}

                # refine
                refine_kpss = polygon.refine_sharpangle_pts(projected_kpss)
                refine_kpss, idx_map = polygon.refine_outside_pts(refine_kpss, camera_cfg.mask_3d, idx_map, inside_pct=0.9)

                # record intermidiate result
                idx_map1 = {j:j+len(kps_3dflatsss) for j in range(len(refine_kpss))}
                idx_map = file.transfer_maps(idx_map, idx_map1)
                res['img_path']=osp.join(customer, city, store, 'car', 'images', day, ch_name, img_name)
                res['vis_path'] = osp.join(customer, city, store, 'car', 'pose', day, 'vis', 'img', ch_name + "_" + save_name)
                res['ch_name'] = ch_name
                res['idx_map'] = idx_map
                res['preds'] = preds
                pred_ress.append(res.copy())

                if refine_kpss == []:
                    continue
                kps_3dflatsss += refine_kpss
                kps_3dflatsss_lid += [ch_name for j in range(len(refine_kpss))]

            #boxes clustering
            groups, idx_map = polygon.polygons_cluster(kps_3dflatsss, kps_3dflatsss_lid, iou_thr=0.3)

            filter_groups, idx_map = polygon.filter_groups_by_area(groups, area_thr=0.6, idx_map=idx_map)
            merge_result, idx_map = polygon.merge_groups(filter_groups, idx_map)

            # rects, ptss, idx_map = polygon.find_minAreaRects(merge_result, idx_map)
            rect_kpss, wheelbases, idx_map = polygon.find_minDistanceRects(merge_result, idx_map)

            refine_kpss, idx_map = polygon.refine_outside_pts(rect_kpss, camera_cfg.mask_3d, idx_map, inside_pct=0.8)
            refine_kpss, idx_map = polygon.filter_overlapping(refine_kpss, idx_map, iou_thr=0.0)
            refine_kpss, idx_map = polygon.filter_too_close_dets(refine_kpss, idx_map)

            kps_2dgroundss = proj.map_3dkpss_to_2dgroundimg(camera_cfg, refine_kpss)
            ## for run single image then output
            kps_2dgroundss, idx_map_tmp, sort_idx_map = file.sort_kps(kps_2dgroundss, idx_map.copy())

            if args.vis_preds == "True" and len(kps_2dgroundss) > 0:
                vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss,
                                    osp.join(post_process_root, "result_in_2d_ground_"+ save_name))

            idx_map1 = {j: j + len(global_kpsss) for j in range(len(refine_kpss))}
            idx_map = file.transfer_maps(idx_map, idx_map1)
            pred_ress = file.update_pred_res_ids(pred_ress, idx_map)
            pred_resss += pred_ress
            if len(refine_kpss)==0:
                continue
            global_kpsss += refine_kpss
            global_kpsss_ld += [i for j in range(len(refine_kpss))]
            used_frame_num += 1

        global_groups, idx_map = polygon.polygons_cluster(global_kpsss, global_kpsss_ld, iou_thr=0.4)
        global_groups, idx_map = polygon.filter_groups_by_object_num(global_groups, used_frame_num * 0.23, idx_map=idx_map)

        merge_result, idx_map = polygon.merge_groups(global_groups, idx_map)
        rect_kpss, wheelbases, idx_map = polygon.find_minDistanceRects(
            merge_result, idx_map, l=polygon.car_fix_length, w=polygon.car_fix_width, final_merge=True)
        refine_kpss, idx_map = polygon.refine_outside_pts(rect_kpss, camera_cfg.mask_3d, idx_map, inside_pct=0.8)
        refine_kpss, idx_map = polygon.filter_overlapping(refine_kpss, idx_map, iou_thr=0.0)
        
        kps_2dgroundss = proj.map_3dkpss_to_2dgroundimg(camera_cfg, refine_kpss)

        # sort results
        kps_2dgroundss, idx_map, sort_idx_map = file.sort_kps(kps_2dgroundss, idx_map)
        pred_resss = file.update_pred_res_ids(pred_resss, idx_map)
        refine_kpss_tmp=refine_kpss.copy()
        refine_kpss = [refine_kpss_tmp[sort_idx_map[i]] for i in range(len(refine_kpss))]
        for res in pred_resss:
            res['pose_3d'] = refine_kpss
        pred_resss = proj.project_3d_bbox_back_to_2d(pred_resss, camera_cfg)

        print("Save rersult to {}".format(pose_root))
        file.save_ress(pred_resss, osp.join(pose_root, "pred_result.json"))
        vis.vis_pred_crops(pred_resss, args.img_root, pose_root)
        file.save_kpss(kps_2dgroundss, osp.join(pose_root, "pose.json"))
        vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, osp.join(pose_root, "pose.jpg"))


def gen_channel_img_dic(img_list):
    ## read to dic
    dic={}
    for line in img_list:
        ch = line.split("_")[3]
        date = line.split("_")[4][:8]
        t = line.split("_")[4][8:8+6]
        key = date + t
        if key not in dic.keys():
            dic[key] = {}
        # if ch in dic[key].keys():
        #     print("repeated ! {}, {}".format(key, ch))
        dic[key][ch] = line
    return dic

if __name__ == "__main__":
    main()
