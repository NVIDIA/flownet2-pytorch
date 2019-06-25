FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server

# Python dependencies
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

#Torch and dependencies:
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision cffi tensorboardX

RUN pip install tqdm scipy scikit-image colorama==0.3.7
RUN pip install setproctitle pytz ipython
RUN pip install requests
