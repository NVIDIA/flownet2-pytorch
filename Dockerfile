FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server

# Python dependencies
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

#Torch and dependencies:
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision cffi tensorboardX==1.6

RUN pip install tqdm==4.32.2 scipy==1.0.0 scikit-image==0.15.0 colorama==0.3.7
RUN pip install setproctitle pytz ipython==7.5.0
RUN pip install requests

# Install correlation model dependencies
RUN bash install.sh

# Dependencies to process flow files
RUN apt-get install ffmpeg
RUN git clone https://github.com/georgegach/flow2image.git /flow2image
