#!/usr/bin/env python
"""
Functions for file io.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import json
import os
import numpy as np
import os.path as osp
import torch

def load_lines(filepath):
    """
    Read a file and return its content as list of lines.

    Input
      filepath  -  input path, string

    Output
      lines     -  lines, 1 x n (list)
    """
    fio = open(filepath, 'r')
    lines = fio.read().splitlines()
    fio.close()

    return lines

def save_lines(filepath, lines, subx='\n'):
    """
    Write a list of line into a file.

    Input
      filepath  -  output path, string
      lines     -  lines, 1 x n (list)
      subx      -  subfix of each line, {None} | '\n' | ...
    """
    assert filepath.__class__ == str

    # create fold if not exist
    fold = osp.dirname(filepath)
    if fold != '' and not osp.exists(fold):
        os.makedirs(fold)

    fio = open(filepath, 'w')
    for line in lines:
        try:
            fio.write(line)
        except UnicodeEncodeError:
            fio.write(line.encode('utf8'))

        if subx is not None:
            fio.write(subx)
    fio.close()

def save_kpss(kpss, path, ids=[]):
    output=[]
    i = 0
    for kps in kpss:
        data = {}
        data['cords'] = np.array(kps)[:,:2].tolist()
        id=i+1
        if len(ids)>0:
            id = ids[i]
        data['id'] = id
        output.append(data)
        i += 1

    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    with open(path, 'w') as outfile:
        json.dump(output, outfile, indent=4)

def load_kpss(path):
    kpss = []
    ids = []
    infos=json.load(open(path, 'r'))

    for info in infos:
        kps = info['cords']
        kps[0].append(1)
        kps[1].append(1)
        kps[2].append(1)
        kps[3].append(1)
        kpss.append(kps)
        ids.append(info['id'])
    return kpss, ids


def save_ress(ress, path):
    images=[]
    for res in ress:

        img_path = res['img_path']
        preds = res['preds']
        boxes = preds.bbox.to(torch.int64).tolist()
        ids = preds.get_field("idxs").to(torch.int).tolist()

        image = {}
        image['image_path'] = img_path
        image['bbox'] = boxes
        image['ids'] = ids
        images.append(image)
    save_dir = osp.dirname(path)
    if (not osp.exists(save_dir)):
        mkdir = "mkdir -p \"" + save_dir + "\""
        os.system(mkdir)
    with open(path, 'w') as outfile:
        json.dump(images, outfile, indent=4)


def transfer_maps(map1, map2):
    transfer_map={}
    for k, v in map1.items():
        if v in map2.keys():
            transfer_map[k] = map2[v]
        else:
            transfer_map[k] = -1
    return transfer_map

def get_map_values(map0):
    ks = np.array(list(map0.keys()))
    ks = np.sort(ks)
    vs=[]
    for k in ks:
        v = map0[k]
        vs.append(v)
    return vs

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

def sort_kps(kpss, idx_map):
    if len(kpss)==0:
        return [], idx_map, {}
    a = np.array(kpss)
    b = [np.min(pts[:, 0]) for pts in a]
    d = np.argsort(b)
    idx_map1 = {d[i]: i+1 for i in range(len(d))}
    sort_idx_map = {i: d[i] for i in range(len(d))}
    kpss = [a[idx] for idx in d]
    idx_map = transfer_maps(idx_map, idx_map1)
    return kpss, idx_map, sort_idx_map

def update_pred_res_ids(pred_ress, idx_map):
    for res in pred_ress:
        idx_map0 = res['idx_map']
        # ids = res['preds'].get_field("ids").to(torch.int).tolist()
        final_ids = transfer_maps(idx_map0, idx_map)
        res['idx_map'] = final_ids
        res['preds'].add_field('idxs', torch.tensor(get_map_values(final_ids), dtype=torch.int, device='cpu'))
    return pred_ress
