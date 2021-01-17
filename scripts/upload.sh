#!/usr/bin/env bash

source argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--store', default='GACNE/guangzhou/cw')
parser.add_argument('--data_root', default='/root/code/car_localization/data')
parser.add_argument('--date', default='20201001')
parser.add_argument('--hdfs_root', default='/bj/yfzhong/')
EOF

local_pose_fold=${DATA_ROOT}/${STORE}/car/pose/${DATE}
if [ ! -f ${local_pose_fold}/pose.json ]; then
  echo "No pose.json file generated!"
  exit
fi

cd ${local_pose_fold}
[ -d crop_images ] && tar cf crop_images.tar crop_images
[ -d vis ] && mv vis/img images && tar cf images.tar images
[ -d vis ] && mv vis debug && tar cf debug.tar debug
cd /root/code/car_localization/

hdfs_pose=${HDFS_ROOT}/${STORE}/car/pred_pose
hdfs_pose_fold=${hdfs_pose}/${DATE}

hdfscli mkdir ${hdfs_pose_fold}
echo "Files in ${local_pose_fold}:"
echo `ls ${local_pose_fold}`
[ -f ${local_pose_fold}/pose.jpg ] &&
echo hdfscli upload -f  ${local_pose_fold}/pose.jpg ${hdfs_pose_fold}/ &&
hdfscli upload -f  ${local_pose_fold}/pose.jpg ${hdfs_pose_fold}/

[ -f ${local_pose_fold}/pose.json ] &&
echo hdfscli upload -f ${local_pose_fold}/pose.json ${hdfs_pose_fold}/ &&
hdfscli upload -f ${local_pose_fold}/pose.json ${hdfs_pose_fold}/

[ -f ${local_pose_fold}/crop_images.tar ] &&
echo hdfscli upload -f ${local_pose_fold}/crop_images.tar ${hdfs_pose_fold}/ &&
hdfscli upload -f ${local_pose_fold}/crop_images.tar ${hdfs_pose_fold}/

[ -f ${local_pose_fold}/images.tar ] &&
echo hdfscli upload -f ${local_pose_fold}/images.tar ${hdfs_pose_fold}/ &&
hdfscli upload -f ${local_pose_fold}/images.tar ${hdfs_pose_fold}/

[ -f ${local_pose_fold}/debug.tar ] &&
echo hdfscli upload -f ${local_pose_fold}/debug.tar ${hdfs_pose_fold}/ &&
hdfscli upload -f ${local_pose_fold}/debug.tar ${hdfs_pose_fold}/

[ -f ${local_pose_fold}/pred_2d_result.json ] &&
echo hdfscli upload -f ${local_pose_fold}/pred_2d_result.json ${hdfs_pose_fold}/ &&
hdfscli upload -f ${local_pose_fold}/pred_2d_result.json ${hdfs_pose_fold}/

echo hdfscli upload -f image_list.csv ${hdfs_pose_fold}/
hdfscli upload -f image_list.csv ${hdfs_pose_fold}/

echo "Car localization module done."