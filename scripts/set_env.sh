export LC_ALL=C
export PYTHONPATH=$PYTHONPATH:~/develop/tf/tensorpack
export TF_CUDNN_USE_AUTOTUNE=0
export JAVA_HOME=/opt/package/jdk1.8.0_181
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
export HADOOP_HOME=/opt/package/hadoop-2.6.5
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib"
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin


cd ../maskrcnn-benchmark
#git pull origin master:master

MODEL_ROOT="models/4s_shop"
if [ ! -d $MODEL_ROOT ]
then
    mkdir -p $MODEL_ROOT
fi
if [ -d $MODEL_ROOT/res101_kp ]
then
    rm -rf $MODEL_ROOT/res101_kp
    rm -rf $MODEL_ROOT/res101_box
    rm -rf $MODEL_ROOT/res101_pad_kp
fi

version='v1.0'
hdfscli download ${hdfs_root}/yfzhong/car/4s_shop/models/${version}/res101_pad_kp models/4s_shop
hdfscli download ${hdfs_root}/yfzhong/car/4s_shop/models/${version}/res101_box models/4s_shop

cd ../car_localization


if [ ! -d models ]
then
    ln -s ../maskrcnn-benchmark/models .
fi

export PYTHONPATH=lib/:/usr/local/lib/python3.6/dist-packages:../maskrcnn-benchmark/:$PYTHONPATH
