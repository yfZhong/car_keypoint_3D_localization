#!/usr/bin/env python
"""
Set up paths for lib fold.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import os
import os.path as osp
import sys

this_dir = osp.dirname(__file__)
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(this_dir)

maskrcnn_fold = osp.join(this_dir, '../', 'maskrcnn-benchmark')
add_path(maskrcnn_fold)

python_pkg ='/usr/local/lib/python3.6/dist-packages'
add_path(python_pkg)