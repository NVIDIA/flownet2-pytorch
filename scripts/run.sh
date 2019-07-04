#!/bin/bash

echo_status()
{
    echo "############################################################"
    echo "$@"
    echo "############################################################"
  x=2
}

# Running inference on Sintel data
echo_status 'Running inference on Sintel data'

python main.py --inference --model FlowNet2 --save_flow \
--inference_dataset MpiSintelClean \
--inference_dataset_root datasets/sintel/training \
--resume checkpoints/FlowNet2_checkpoint.pth.tar \
--save datasets/sintel/output

# Converting sintel data optical flow files to color coded image files

echo_status 'Converting sintel data optical flow files to color coded image files'

python /flow2image/f2i.py \
datasets/sintel/output/inference/run.epoch-0-flow-field/*.flo \
-o datasets/sintel/output/color_coding

# Generating custom frames from video

echo_status 'Generating custom frames from video'

ffmpeg -i datasets/dancelogue/sample-video.mp4 datasets/dancelogue/frames/output_%02d.png

# Running Inference on the custom video frames

echo_status 'Running Inference on the custom video frames'

python main.py --inference --model FlowNet2 --save_flow \
--inference_dataset ImagesFromFolder \
--inference_dataset_root datasets/dancelogue/frames/ \
--resume checkpoints/FlowNet2_checkpoint.pth.tar \
--save datasets/dancelogue/output

# Generating the color coding images ffor optical flow files for video

echo_status 'Generating the color coding images ffor optical flow files for video'

python /flow2image/f2i.py \
datasets/dancelogue/output/inference/run.epoch-0-flow-field/*.flo \
-o datasets/dancelogue/output/color_coding -v -r 30
