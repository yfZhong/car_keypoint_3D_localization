data='imgs'
for i in $(find ${data}/ -name '*.jpg');
do
    IFS='/'
    read -a paths <<<$i
    IFS='_'
    read -a vals <<<${paths[1]}
    IFS=
    customer="${vals[0]}"
    city="${vals[1]}" 
    store="${vals[2]}"
    channel="${vals[3]}"
    datetime="${vals[4]}"
    frame="${vals[5]}"
    date="${datetime:0:8}"
    time="${datetime:8:6}"
    frame_id=${frame:0:5}

    echo $customer $city $store ${channel} ${date} ${time} ${frame_id}
    real_time=$((${time} + (${frame_id}-1)*100))
    echo ${real_time}

    fold=./${data}"_root"/${customer}/${city}/${store}/car/images/$date/${channel}
    echo $fold
    image_name=${date}${real_time}0000.jpg
    if [ ! -d fold ]
    then
        mkdir -p $fold
    fi
    origin_image=./${data}/${customer}_${city}_${store}_${channel}_${datetime}_${frame}
    echo ${origin_image}
    cp ${origin_image} ${fold}/${image_name}
done
