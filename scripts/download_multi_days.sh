#!/usr/bin/env bash


if [ -f "program_id_time.txt" ]
then
    rm program_id_time.txt
fi

download(){
    IFS=$'\t'
    read -a customer <<<$1
    read -a city <<<$2
    read -a store <<<$3
    read -a start_day <<<$4
    read -a end_day <<<$5
    #day=$(date  +"%Y%m%d")
    #day=$(date  +"%Y%m%d" -d  "- 1 days")
    i=0
    while true; do
	day=$(date -d "${start_day}  +${i} days" "+%Y%m%d")
	echo $day
        if [ ${day} -ge ${end_day} ];then
            break
        fi
    
        echo $customer $city $store $day

        fold=${hdfs_root}/customer/${customer}/${city}/${store}/car/images/$day
        #hdfs dfs -test -e ${fold}/_done_
        #if [ ! $? == 0 ];
        #then
	#    echo "No available image data!"
	#    exit
        #fi
        local_images_fold=./hdfs_data/${customer}/${city}/${store}/car/images
        local_fold=${local_images_fold}/$day
        if [ ! -d ${local_fold} ]
        then
    	    mkdir -p ${local_images_fold}
	    hdfscli download ${fold} ${local_images_fold}/
        fi
        if [ ! -d ${local_fold} ]
        then
	    echo "No available image data!"
	    exit
        fi
        echo ${customer}$'\t'${city}$'\t'${store}$'\t'${day}$'\t'130000 >>'program_id_time.txt'
        echo "Download data: "$fold

	i=$((i+1))
    done
}

export -f download

cat "program_id.txt" | xargs -I {} bash -c "download {}"
