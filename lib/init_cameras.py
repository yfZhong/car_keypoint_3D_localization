#!/usr/bin/env python
"""
Functions for  loading camera calibration.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import os.path as osp
import cv2
from easydict import EasyDict as ED
import _init_paths
import camera_info_packet
import json
import proj
import yaml

def load_cameras(db = 'nanling', camerainfo_root = None, ch_names=None):
    '''
    :param db: database name
    :param camerainfo_root: path to camerainfo root
    :param ch_names: camera channel names
    :return: camera_cfg
    '''
    camera_cfg = ED()
    cameras = {}
    for ch_name in ch_names:
        # camera info
        camera = ED()
        # camera.ch_name = ch_name
        camera.cam = camera_info_packet.get_camera_info_by_name(
            osp.join(camerainfo_root, 'regular'), ch_name)

        cameras[ch_name] = camera
    camera_cfg.cameras = cameras
    camera_cfg.global_camera = load_global_calibration(osp.join(camerainfo_root, 'floorinfos'))
    camera_cfg.mask_3d = load_indoor_region(camera_cfg.global_camera, camerainfo_root)
    return camera_cfg

def load_indoor_region(camera, camerainfo_root):
    entrance_info_path = osp.join(camerainfo_root, 'entrance_info.yaml')
    # fs = cv2.FileStorage(entrance_info_path, cv2.FILE_STORAGE_READ)
    with open(entrance_info_path) as file:
        dic = yaml.load(file)
        indoor_region = dic['indoor_region']
        pt_3ds =[]
        for pt_2d in indoor_region:
            pt_3d = proj.project_2dground_to_3dflat(camera, pt_2d)
            pt_3ds.append(pt_3d)
    return pt_3ds

def load_camera_mask(path):
    mask = json.load(open(path))
    return mask
def get_3d_mask(cameras, mask2d):
    mask3d={}
    for ch_name, mask in mask2d.items():
        mask3d[ch_name] = proj.project_2ds_to_3dflat(cameras[ch_name], mask)
    return mask3d

def load_global_calibration(floor_root):
    '''
    :param floor_root:  floor map root folder
    :return: global camera info
    '''
    camera = ED()

    # global
    # 2dmap <-> 3dmap
    floormap_path = osp.join(floor_root, 'floor_map.yml')
    fs = cv2.FileStorage(floormap_path, cv2.FILE_STORAGE_READ)
    camera.affine_mat = fs.getNode("affine_matrix").mat()
    camera.inv_affine_mat = fs.getNode("inv_affine_matrix").mat()
    camera.H_3to2map = fs.getNode("H_3to2map").mat()
    camera.H_2mapto3 = fs.getNode("H_2mapto3").mat()
    floorimg_path = osp.join(floor_root, 'floor.jpg')
    floorimg = cv2.imread(floorimg_path)
    camera.floorimg = floorimg
    fs.release()

    return camera

