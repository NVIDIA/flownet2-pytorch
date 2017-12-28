#!/bin/bash
git clone https://github.com/lmb-freiburg/flownet2-docker
cd flownet2-docker

sed -i '$ a RUN apt-get update && apt-get install -y python-pip \
RUN pip install --upgrade pip \
RUN pip install numpy -I \
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl \
RUN pip install cffi ipython' Dockerfile

make

