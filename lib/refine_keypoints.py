#!/usr/bin/env python
"""
Functions for refine keypoints.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import cv2
import numpy as np

def CalcEuclideanDistance(point1,point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance

def CalcFourthPoint(point1,point2,point3):
    D = (point1[0]+point2[0]-point3[0],point1[1]+point2[1]-point3[1])
    return D

def JudgeBeveling(point1,point2,point3):
    dist1 = CalcEuclideanDistance(point1,point2)
    dist2 = CalcEuclideanDistance(point1,point3)
    dist3 = CalcEuclideanDistance(point2,point3)
    dist = [dist1, dist2, dist3]
    max_dist = dist.index(max(dist))
    if max_dist == 0:
        D = CalcFourthPoint(point1,point2,point3)
    elif max_dist == 1:
        D = CalcFourthPoint(point1,point3,point2)
    else:
        D = CalcFourthPoint(point2,point3,point1)
    return D