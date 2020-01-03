#!/bin/bash
INPUT_FOLDER="${1}"

sudo nvidia-docker build -t flownet2-pytorch:genflow .

sudo nvidia-docker run --rm -ti \
    --user=${UID}${GID+:}${GID} \
    --volume="${INPUT_FOLDER}":/datasets:rw \
    --workdir=/flownet2-pytorch \
    --ipc=host \
    flownet2-pytorch:genflow \
    /datasets "${@:2}"

