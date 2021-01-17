#!/usr/bin/env bash


source argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--store', default="GACNE/guangzhou/cw")
parser.add_argument('--data_dir', default='/root/code/car_localization/data')
parser.add_argument('--start_time', default='130000')
parser.add_argument('--date', default='20201001')
EOF


#if [ -z "$day" ]; then
#    day=$(date  +"%Y%m%d")
#    #day=$(date  +"%Y%m%d" -d  "- 1 days")
#fi
#echo $customer $city $store $day

fold=/bj/yfzhong/${STORE}/car/images/${DATE}


rm -rf ${local_fold}/* || true
mkdir -p ${local_fold} || true

echo hdfscli download ${fold}/_done_ ${local_fold}/
hdfscli download ${fold}/_done_ ${local_fold} || true
if [ ! -f ${local_fold}/_done_ ]
then
  echo "No date done signal detected. Images have not been prepared!"
  exit
fi
echo hdfscli download -f ${fold}/ch* ${local_fold}/
hdfscli download -f ${fold}/ch* ${local_fold}/
#echo ${customer}$'\t'${city}$'\t'${store}$'\t'${day}$'\t'${start_time} >>'program_id_time.txt'
echo "Download data to : "$fold
