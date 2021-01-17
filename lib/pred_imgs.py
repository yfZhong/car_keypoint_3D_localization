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
import yaml
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)

def init_model(config_file, conf_thr, min_image_size, model_root, opts):
    # load model config from file and command-line arguments
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=conf_thr,
        min_image_size=min_image_size,
        model_root=model_root
    )
    return coco_demo

def load_parameters(site_id):
    default_config_path = "config/default.yaml"
    with open(default_config_path) as parameters:
        params = yaml.safe_load(parameters)
    store_config_path = osp.join("config", site_id, "config.yaml")
    if os.path.exists(store_config_path):
        with open(store_config_path) as parameters:
            store_params = yaml.safe_load(parameters)
        # params.update(store_params)
        params = update_dict(params, store_params)
    return params

def update_dict(dic1, dic2):
    for i in dic2:
        if i in dic1:
            if type(dic1[i]) is dict and type(dic2[i]) is dict:
                dic1[i] = update_dict(dic1[i], dic2[i])
            else:
                dic1[i] = dic2[i]
        else:
            dic1[i] = dic2[i]
    return dic1

def get_old_camera_info_root(store, standard_camera_info_root, day):
    if store == "qybc" and int(day) < 20190929:
        standard_camera_info_root += "_before_20190929"
    if store == "hddb" and int(day) < 20191103:
        standard_camera_info_root += "_before_20191103"
    if store == "bydz" and int(day) < 20191209:
        standard_camera_info_root += "_before_20191209"
    elif store == "bydz" and int(day) < 20191222:
        standard_camera_info_root += "_before_20191222"
    return standard_camera_info_root

def main():
    args = parse_args()

    # load model config from file and command-line arguments
    box_model = init_model(args.config_file_box, args.confidence_threshold, args.min_image_size, args.model_root, args.opts)
    kp_model = init_model(args.config_file_kp, args.confidence_threshold, args.min_image_size_kp, args.model_root, args.opts)

    site_id = args.store
    day = args.date
    start_time = args.start_time[:6]
    logging.info("{}\t{}\t{}".format(site_id, day, start_time))
    params = load_parameters(site_id)
    camera_info_root = osp.join(args.camera_info_dir, site_id)

    # load camera config
    ch_names = os.listdir(osp.join(camera_info_root, "regular"))
    camera_cfg = load_cameras(camerainfo_root=camera_info_root, ch_names=ch_names)

    if args.vis_gts=="True" and args.gt_annos is not None:
        dic_img_gts = boxlist_process.load_img_gts_boxlist(args.gt_annos)

    img_path_root = osp.join(args.img_root, site_id, 'car', 'images', day)
    # vis_root = osp.join(args.save_root, site_id, 'car', 'vis', day)
    pose_root = osp.join(args.save_root, site_id, 'car', 'pose', day)
    vis_root = osp.join(pose_root, 'vis')
    post_process_root = osp.join(vis_root, 'post_process')
    global_kpsss = []
    global_kpsss_ld = []
    # start_time=start_time[:6]
    num_frames = params['multi_frame_merge']['num_frames']
    final_save_name = start_time + ".jpg"
    used_frame_num = 0
    pred_resss = []

    for i in range(num_frames):
        res={}
        t1 = str(int(start_time)+(i*100)).zfill(6)
        kps_3dflatsss = []
        kps_3dflatsss_lid = []
        img_name_tmp = day + t1[:4] + "*.jpg"
        # img_name_tmp = day + "*.jpg"
        save_name = t1 +"_"+str(i)+ ".jpg"
        pred_ress=[]
        num_images=0
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
            num_images += 1
            if args.vis_gts=="True":
                img_key = osp.join(site_id, 'car', 'images', day, ch_name, img_name)
                if img_key not in dic_img_gts.keys():
                    continue
                gts = dic_img_gts[img_key]
                preds = gts
            else:
                #do img prediction
                cfg.merge_from_file(args.config_file_box)
                composite, preds = box_model.run_on_opencv_image(img)
                cfg.merge_from_file(args.config_file_kp)
                # preds = pred_keypoints.get_boxes_kps(img, preds, kp_model)
                box_score_thr = params['single_frame']['box_score_thr']
                try:
                    box_score_thr = params['single_frame'][ch_name]['box_score_thr']
                except:
                    pass

                preds = pred_keypoints.get_boxes_kps_pad(img, preds, kp_model, score_thr=box_score_thr)
                # if preds != None:
                #     vis.vis_kps_in_img(img, preds, osp.join(vis_root, "img0", ch_name + save_name))

            box_height_thr = params['single_frame']['box_height_thr']  #gt  200 pred  250
            box_width_thr = params['single_frame']['box_width_thr']  #gt  200 pred  250
            area_thr = params['single_frame']['area_thr'] # gt  1000 pred  10000
            try:
                box_height_thr = params['single_frame'][ch_name]['box_height_thr']
                box_width_thr = params['single_frame'][ch_name]['box_width_thr']
                area_thr = params['single_frame'][ch_name]['area_thr']
            except:
                pass

            preds = boxlist_process.filter_preds(preds, box_height_thr, box_width_thr, area_thr)

            preds = proj.add_3d_keypoints(preds, camera_cfg, ch_name, args.vis_gts)
            if preds == None:
                continue

            projected_kpss = preds.get_field('keypoints_3d').to(torch.float).tolist()
            idxs = preds.get_field('idxs').to(torch.int).tolist()
            idx_map = {j:j for j in idxs}

            # refine
            # refine_kpss = polygon.refine_lowscore_pts(projected_kpss)
            refine_kpss = polygon.refine_sharpangle_pts(projected_kpss)
            # refine_kpss = polygon.refine_triangle_pts(refine_kpss)
            inside_pct = params['single_frame']['inside_pct']
            refine_kpss, idx_map = polygon.refine_outside_pts(refine_kpss, camera_cfg.mask_3d, idx_map, inside_pct=inside_pct)
            # record intermidiate result
            idx_map1 = {j:j+len(kps_3dflatsss) for j in range(len(refine_kpss))}
            idx_map = file.transfer_maps(idx_map, idx_map1)
            res['img_path']=osp.join(site_id, 'car', 'images', day, ch_name, img_name)
            res['vis_path'] = osp.join(site_id, 'car', 'pose', day, 'vis', 'img', ch_name + "_" + save_name)
            res['ch_name'] = ch_name
            res['idx_map'] = idx_map
            res['preds'] = preds
            pred_ress.append(res.copy())

            if refine_kpss == []:
                continue
            kps_3dflatsss += refine_kpss
            kps_3dflatsss_lid += [ch_name for j in range(len(refine_kpss))]
            # if args.vis_preds=="True":
                #vis.vis_kps_in_img(img, preds0, osp.join(vis_root, "img0", ch_name + save_name))
                # vis.vis_kps_on_blank_img(projected_kpss, osp.join(vis_root, "proj", ch_name +"_"+ save_name),  camera_cfg.mask_3d)
                # vis.vis_kps_on_blank_img(refine_kpss, osp.join(vis_root, "refine", ch_name +"_"+ save_name))

        logging.info("frame_id: {}, number images: {}, detected bboxs: {}".format(i, num_images, len(kps_3dflatsss)))
        # merge single frame
        #boxes clustering
        iou_thr = params['single_frame_merge']['iou_thr']
        groups, idx_map = polygon.polygons_cluster(kps_3dflatsss, kps_3dflatsss_lid, iou_thr=iou_thr)
        area_thr = params['single_frame_merge']['area_thr']
        filter_groups, idx_map = polygon.filter_groups_by_area(groups, area_thr=area_thr, idx_map=idx_map)
        merge_result, idx_map = polygon.merge_groups(filter_groups, idx_map)

        # rects, ptss, idx_map = polygon.find_minAreaRects(merge_result, idx_map)
        rect_kpss, wheelbases, idx_map = polygon.find_minDistanceRects(merge_result, idx_map)

        inside_pct = params['single_frame_merge']['inside_pct']
        refine_kpss, idx_map = polygon.refine_outside_pts(rect_kpss, camera_cfg.mask_3d, idx_map, inside_pct=inside_pct)
        refine_kpss, idx_map = polygon.filter_overlapping(refine_kpss, idx_map, iou_thr=0.0)
        refine_kpss, idx_map = polygon.filter_too_close_dets(refine_kpss, idx_map)

        kps_2dgroundss = proj.map_3dkpss_to_2dgroundimg(camera_cfg, refine_kpss)

        # kps_2dgroundss = file.load_kpss(osp.join(pose_root, t[:6] + ".json"))
        ## for run single image then output
        kps_2dgroundss, idx_map_tmp, sort_idx_map = file.sort_kps(kps_2dgroundss, idx_map.copy())
        # kps_2dgroundss, idx_map, sort_idx_map = file.sort_kps(kps_2dgroundss, idx_map)
        # refine_kpss_tmp = refine_kpss.copy()
        # refine_kpss = [refine_kpss_tmp[sort_idx_map[i]] for i in range(len(refine_kpss))]

        if args.vis_preds == "True" and len(kps_2dgroundss) > 0:
            # pred_ress_tmp = file.update_pred_res_ids(pred_ress, idx_map_tmp)
            # vis.vis_pred_ress(pred_ress_tmp, args.img_root, args.save_root)

            vis.vis_kps_on_blank_img(kps_3dflatsss, osp.join(post_process_root, "1_kps_in_3d_" + save_name), camera_cfg.mask_3d)
            # vis.vis_kps_on_blank_img(kps_3dflatsss, osp.join(post_process_root, "projected_" + save_name))
            vis.vis_group(groups, osp.join(post_process_root, "2_cluster_"+ save_name))
            vis.vis_group(filter_groups, osp.join(post_process_root, "3_filter_cluster_"+save_name))
            vis.vis_kps_on_blank_img(merge_result,
                                     osp.join(post_process_root, "4_1_merge_each_cluster_" +save_name), camera_cfg.mask_3d)
            vis.vis_kps_on_blank_img(merge_result + rect_kpss,
                                     osp.join(post_process_root, "4_2_merge_each_cluster_with_rect_" + save_name),
                                     camera_cfg.mask_3d)
            vis.vis_kps_on_blank_img(refine_kpss, osp.join(post_process_root, "4_3_rect_"+ save_name), camera_cfg.mask_3d)
            vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss,
                                osp.join(post_process_root, "5_result_in_2d_ground_"+ save_name))
            # vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, osp.join(pose_root, save_name))
            # file.save_ress(pred_ress, osp.join(pose_root, "result.json"))

        ##run single image then output
        # file.save_kpss(kps_2dgroundss, osp.join(pose_root, t[:6] + ".json"))
        # vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, osp.join(pose_root, t[:6] + ".jpg"))
        # file.save_kpss(kps_2dgroundss, osp.join(pose_root, "pose.json"))
        # vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, osp.join(pose_root, "pose.jpg"))
        # break

        idx_map1 = {j: j + len(global_kpsss) for j in range(len(refine_kpss))}
        idx_map = file.transfer_maps(idx_map, idx_map1)
        pred_ress = file.update_pred_res_ids(pred_ress, idx_map)
        pred_resss += pred_ress
        if len(refine_kpss)==0:
            continue
        global_kpsss += refine_kpss
        global_kpsss_ld += [i for j in range(len(refine_kpss))]
        used_frame_num += 1

    # merge multi frames
    iou_thr = params['multi_frame_merge']['iou_thr']
    global_groups, idx_map = polygon.polygons_cluster(global_kpsss, global_kpsss_ld, iou_thr=iou_thr)
    used_frame_pct = params['multi_frame_merge']['used_frame_pct']
    global_groups, idx_map = polygon.filter_groups_by_object_num(global_groups, used_frame_num * used_frame_pct, idx_map=idx_map)

    # filter_groups, idx_map = polygon.filter_groups_by_area(global_groups, area_thr=0.2, idx_map=idx_map)
    merge_result, idx_map = polygon.merge_groups(global_groups, idx_map)
    # rects, ptss, idx_map = polygon.find_minAreaRects(merge_result, idx_map)
    rect_kpss, wheelbases, idx_map = polygon.find_minDistanceRects(
        merge_result, idx_map, l=polygon.car_fix_length, w=polygon.car_fix_width, final_merge=True)
    # rect_kpss = polygon.refine_size_use_fix_lw(rect_kpss)
    # rect_kpss = polygon.refine_size_according_to_wheelbase(rect_kpss, wheelbases, pct=1)
    inside_pct = params['multi_frame_merge']['inside_pct']
    refine_kpss, idx_map = polygon.refine_outside_pts(rect_kpss, camera_cfg.mask_3d, idx_map, inside_pct=inside_pct)
    overlap_iou_thr = params['multi_frame_merge']['overlap_iou_thr']
    refine_kpss, idx_map = polygon.filter_overlapping(refine_kpss, idx_map, iou_thr=overlap_iou_thr)

    kps_2dgroundss = proj.map_3dkpss_to_2dgroundimg(camera_cfg, refine_kpss)

    # sort results
    kps_2dgroundss, idx_map, sort_idx_map = file.sort_kps(kps_2dgroundss, idx_map)
    pred_resss = file.update_pred_res_ids(pred_resss, idx_map)

    refine_kpss_tmp=refine_kpss.copy()
    refine_kpss = [refine_kpss_tmp[sort_idx_map[i]] for i in range(len(refine_kpss))]
    for res in pred_resss:
        res['pose_3d'] = refine_kpss
    pred_resss = proj.project_3d_bbox_back_to_2d(pred_resss, camera_cfg)
    # pred_resss = proj.project_kps3d_back_to_2d(pred_resss, camera_cfg)

    # load if pose.json be modified
    # kps_2dgroundss, ids = file.load_kpss(osp.join(pose_root, "pose.json"))

    if args.vis_preds == "True":
        vis.vis_pred_ress(pred_resss, args.img_root, args.save_root)
        if len(kps_2dgroundss) >0:
            vis.vis_kps_on_blank_img(global_kpsss, osp.join(post_process_root, "6.0_boxes_in_3d.jpg"), camera_cfg.mask_3d)
            vis.vis_group(filter_groups, osp.join(post_process_root, "6.1_final_cluster.jpg"))
            vis.vis_kps_on_blank_img(merge_result + refine_kpss,
                                 osp.join(post_process_root, "6.2_final_3d_rects.jpg"), camera_cfg.mask_3d)
            vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss,
                            osp.join(post_process_root, "7_final_result_in_2d_ground.jpg"))

    logging.info("Save rersult to {}".format(pose_root))
    file.save_ress(pred_resss, osp.join(pose_root, "pred_2d_result.json"))
    vis.vis_pred_crops(pred_resss, args.img_root, pose_root)
    # file.save_kpss(refine_kpss, osp.join(pose_root, t[:6] + "_3d.json"))
    # file.save_kpss(kps_2dgroundss, osp.join(pose_root, t[:6] + ".json"))
    file.save_kpss(kps_2dgroundss, osp.join(pose_root, "pose.json"))
    vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, osp.join(pose_root, "pose.jpg"))
    vis.vis_2dgroundimg(camera_cfg.global_camera.floorimg.copy(), kps_2dgroundss, osp.join(pose_root, final_save_name))


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
