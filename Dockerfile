FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server python-pip

RUN pip install --upgrade pip

RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
RUN pip install torchvision cffi tensorboardX

RUN pip install tqdm scipy scikit-image colorama==0.3.7 
RUN pip install setproctitle pytz ipython