#!/usr/bin/env python
"""
Functions for camera projections.

History
  create  -  Yongfeng Zhong (yongfeng_zhong@hotmail.com), 2019-10
"""

import yaml
import numpy as np
import cv2
import os

def get_transition_matrix(tvec, rvec):
    rot_mat, _ = cv2.Rodrigues(rvec)
    trans_mat = np.zeros((3, 4), rot_mat.dtype)
    trans_mat[0:3, 0:3] = rot_mat[:]
    trans_mat[:3, 3] = np.transpose(tvec[:])
    return trans_mat

# only support wcc
def get_camera_info_by_name(calib_root, name):

    camera_info = None
    intrinsic_yaml_path = os.path.join(calib_root, name, 'intrinsic-calib.yml')
    extrinsic_yaml_path = os.path.join(calib_root, name, 'extrinsic-calib.yml')
    camera_info = CameraInfoPacket(intrinsic_yaml_path, extrinsic_yaml_path)
    return camera_info


class CameraInfoPacket(object):

    def __init__(self, intrinsic_calibration_path, extrinsic_calibration_path):
        self._intrinsic_calibration_path = intrinsic_calibration_path
        self._extrinsic_calibration_path = extrinsic_calibration_path

        # intrinsic params
        self._camera_matrix = None
        self._image_width, self._image_height = None, None
        self._distortion_coefficients = None

        # extrinsic params
        self._rvec, self._tvec = None, None

        # projection matrix
        self._mat_rt = None
        self._mat_p = None
        self.load(intrinsic_calibration_path, extrinsic_calibration_path)

    @property
    def image_height(self):
        return self._image_height

    @property
    def image_width(self):
        return self._image_width

    @property
    def mat_p(self):
        return self._mat_p

    def load(self, intrinsic_calibration_path, extrinsic_calibration_path):
        # a helper function to load opencv matrix in yaml
        def opencv_matrix(loader, node):
            mapping = loader.construct_mapping(node, deep=True)
            mat = np.array(mapping["data"])
            mat.resize(mapping["rows"], mapping["cols"])
            return mat
        yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

        # load intrinsic_calibration_path
        skip_lines = 2
        with open(intrinsic_calibration_path) as intrinsic_file:
            for i in range(skip_lines):
                _ = intrinsic_file.readline()
            intrinsic_params = yaml.load(intrinsic_file)

        self._camera_matrix = intrinsic_params['camera_matrix']
        self._image_width, self._image_height = intrinsic_params['image_width'], intrinsic_params['image_height']
        self._distortion_coefficients = intrinsic_params['distortion_coefficients']

        # load extrinsic_calibration_path
        skip_lines = 2
        with open(extrinsic_calibration_path) as extrinsic_file:
            for i in range(skip_lines):
                _ = extrinsic_file.readline()
            extrinsic_params = yaml.load(extrinsic_file)

        self._rvec = extrinsic_params['rvec']
        self._tvec = extrinsic_params['tvec']

        self._mat_rt = get_transition_matrix(self._tvec, self._rvec)
        self._mat_p = self._camera_matrix.dot(self._mat_rt)

    def undistort_point(self, point2d):
        points2d = np.array([[point2d.tolist()]], dtype=np.float64)
        return cv2.undistortPoints(points2d, self._camera_matrix, self._distortion_coefficients, P=self._camera_matrix)[0][0]

    # project 2d points to 3d
    def project_3d_given_z(self, point2d, z, should_undistort=False, with_prob=False):
        if should_undistort:
            point2d = self.undistort_point(point2d)

        x, y = point2d[0], point2d[1]
        matrix_a = np.zeros((3, 3), dtype=np.float64)
        matrix_a[:3, :2] = self._mat_p[:3, :2]
        matrix_a[:3, 2] = np.array([-x, -y, -1.0])

        vector_b = np.array([-(z * self._mat_p[0, 2] + self._mat_p[0, 3]),
                             -(z * self._mat_p[1, 2] + self._mat_p[1, 3]),
                             -(z * self._mat_p[2, 2] + self._mat_p[2, 3])])
        vec_xy = np.linalg.inv(matrix_a).dot(np.transpose(vector_b))
        obj_point = [vec_xy[0], vec_xy[1], z]
        if not with_prob:
            return obj_point
        else:
            probe_points = self.project_3d_given_z(point2d - [0., 1.], 0, should_undistort=should_undistort)
            varation = np.linalg.norm(np.array(obj_point) - np.array(probe_points), axis=0)
            C = 7.0
            prob = np.exp(-C*varation)
            return obj_point, prob

    # def project_3d_to_2d(self, points3d):
    #     points2d = []
    #     for point3d in points3d:
    #         point_vec = np.array(
    #             [point3d[0], point3d[1], point3d[2], 1.0], dtype=np.float64)
    #         proj_2d = self._mat_p.dot(point_vec)
    #         proj_2d = [proj_2d[0]/proj_2d[2], proj_2d[1]/proj_2d[2]]
    #         proj_2d_distorted = self.distort_point(proj_2d)
    #         points2d.append(proj_2d_distorted)
    #     return points2d
    #
    # def distort_point(self, point2d):
    #     fx = self._camera_matrix[0, 0]
    #     fy = self._camera_matrix[1, 1]
    #     cx = self._camera_matrix[0, 2]
    #     cy = self._camera_matrix[1, 2]
    #     k1 = self._distortion_coefficients[0, 0]
    #     k2 = self._distortion_coefficients[0, 1]
    #     p1 = self._distortion_coefficients[0, 2]
    #     p2 = self._distortion_coefficients[0, 3]
    #     k3 = self._distortion_coefficients[0, 4]
    #
    #     x = (point2d[0] - cx) / fx
    #     y = (point2d[1] - cy) / fy
    #
    #     r2 = x * x + y * y
    #     x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    #     y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    #     x_distort = x_distort * fx + cx
    #     y_distort = y_distort * fy + cy
    #
    #     return [x_distort, y_distort]

    def project_3d_to_2d(self, points3d):
        obj_points = np.array([points3d], dtype=np.float32)
        img_points, _ = cv2.projectPoints(obj_points, self._rvec, self._tvec, self._camera_matrix, self._distortion_coefficients)
        return img_points.reshape(img_points.shape[0],img_points.shape[2])

