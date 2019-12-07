#!/bin/bash
INPUT_FOLDER="${1}"
OUTPUT_FOLDER="${2}"

sudo nvidia-docker build -t flownet2-pytorch:flow_runner .

if [ -z "${OUTPUT_FOLDER}" ]; then
  sudo nvidia-docker run --rm -ti \
    --volume="${INPUT_FOLDER}":/datasets:rw \
    --workdir=/flownet2-pytorch \
    --ipc=host \
    flownet2-pytorch:flow_runner \
    /datasets
else
  sudo nvidia-docker run --rm -ti \
    --volume="${INPUT_FOLDER}":/datasets:rw \
    --volume="${OUTPUT_FOLDER}":/output
    --workdir=/flownet2-pytorch \
    --ipc=host \
    flownet2-pytorch:flow_runner \
    /datasets \
    /output
fi

