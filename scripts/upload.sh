#!/usr/bin/env bash

upload(){
    IFS=$'\t'
    read -a customer <<<$1
    read -a city <<<$2
    read -a store <<<$3
    read -a day <<<$4
    read -a time <<<$5
    time="${time:0:6}"
    
    #echo $customer $city $store $day $time    
    local_pose_fold=./hdfs_data/${customer}/${city}/${store}/car/pose/$day
    if [ ! -f ${local_pose_fold}/pose.json ];
    then
	echo "No pose.json file generated!"
	exit
    fi
    cd ${local_pose_fold}
    tar cvf crop_images.tar crop_images
    tar cvf vis.tar vis
    cd /root/code/car_localization/

    hdfs_pose=${hdfs_root}/customer/${customer}/${city}/${store}/car/pred_pose
    hdfs_pose_fold=${hdfs_pose}/$day
    #hadoop fs -test -d ${hdfs_pose_fold}
    #if [ $? == 0 ]; then
    #    #echo "exists"
    #    hdfscli delete -r ${hdfs_pose_fold}
    #fi
    hdfscli delete -rf ${hdfs_pose_fold}
    hdfscli mkdir ${hdfs_pose_fold}
    hdfscli upload ${local_pose_fold}/pose.jpg ${hdfs_pose_fold}/
    hdfscli upload ${local_pose_fold}/pose.json ${hdfs_pose_fold}/
    hdfscli upload ${local_pose_fold}/crop_images.tar ${hdfs_pose_fold}/
    hdfscli upload ${local_pose_fold}/vis.tar ${hdfs_pose_fold}/
    echo "upload" ${local_pose_fold} "to" ${hdfs_pose_fold}


}
export -f upload

cat "program_id_time.txt" | xargs -I {} bash -c "upload {}"
