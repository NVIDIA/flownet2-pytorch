#!/bin/bash

sudo nvidia-docker build -t $USER/flownet2:latest .
sudo nvidia-docker run --rm -ti --volume=$(pwd):/flownet2-pytorch:rw --workdir=/flownet2-pytorch --ipc=host $USER/flownet2:latest /bin/bash
