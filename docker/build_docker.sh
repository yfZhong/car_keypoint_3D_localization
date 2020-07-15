#!/usr/bin/env bash
set -e

TAG=docker.cn/yfzhong/car_keypoint_3D_localization:0.0.1
docker build . -t $TAG -f Dockerfile
docker push $TAG
