ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN mkdir /root/code
ADD  car_keypoint_3D_localization /root/code/car_keypoint_3D_localization
ADD car_keypoint_3D_localization/maskrcnn-benchmark /root/code/maskrcnn-benchmark
ADD apex /root/code/apex
COPY ssh.tar /etc

RUN rm -rf /etc/ssh && tar xf ssh.tar && rm -f ssh.tar && \
    cd /root/code/apex && python setup.py install --cuda_ext --cpp_ext && \
    cd /root/code/maskrcnn-benchmark && python setup.py build develop && \

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas xgboost taskflow psutil paramiko shapely oss2 pyyaml tqdm smart_open dask pyspark pyhive thrift

RUN apt-get update && /bin/echo -e "yes\n" | apt-get install openssh-server

WORKDIR /root/code/car_keypoint_3D_localization
RUN chmod -R a+w .
ENV PYTHONPATH lib/:/usr/local/lib/python3.6/dist-packages:/root/code/maskrcnn-benchmark/:$PYTHONPATH


