#!/bin/bash
sudo nvidia-docker build -t flownet2-pytorch:CUDA9-py35 .
sudo nvidia-docker run --rm -ti --volume=$(pwd):/flownet2-pytorch:rw --workdir=/flownet2-pytorch --ipc=host flownet2-pytorch:CUDA9-py35 /bin/bash
