import torch
import torch.nn as nn
import numpy as np
import os
import models
import time
import argparse, os, sys, subprocess
import matplotlib.pyplot as plt
from PIL import Image
from utils import flow_utils

"""
Test functionality by running flownet-2 on a pair of images provided in 
./test_images Download the FN2 weight file and put it in ./model_weights 
dir (or create) a new dir and pass it as argument.
"""
def load_sequence(img1, img2):
    # Load images
    leftimage = Image.open(img1)
    rightimage = Image.open(img2)

    images = [leftimage, rightimage]

    frame_size = images[0].size[:2]
    # resize images to the nearest multiple of 64
    render_size = list(frame_size)
    render_size[0] = ((frame_size[0]) // 64) * 64
    render_size[1] = ((frame_size[1]) // 64) * 64
    for i in range(len(images)):
        images[i] = images[i].resize((render_size[0], render_size[1]))
        images[i] = np.asarray(images[i]).astype(np.float32)

    images_tensor = np.array(images).transpose(3, 0, 1, 2)
    images_tensor = torch.from_numpy(images_tensor.astype(np.float32))
    # Add a dimension in the beginning for batch size
    images_tensor = images_tensor.unsqueeze(0)
    return images_tensor, images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_max", type=float, default = 255.)
    parser.add_argument('--weight_file', default='model_weights/FlowNet2_checkpoint.pth.tar', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--img1', default='test_images/frame_0003.png', type=str,
                        metavar='PATH', help='path to left stereo image')
    parser.add_argument('--img2', default='test_images/frame_0006.png', type=str,
                        metavar='PATH', help='path to right stereo image')
    parser.add_argument('--mode',
                    default='torch',
                    const='torch',
                    nargs='?',
                    choices=['cuda', 'cpp', 'torch'],
                    help='mode cuda, cpp, or torch (default: %(default)s)')

    # These arguments are used within the nvidia flownet models.
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    
    
    args = parser.parse_args()
    print ("Running in {} mode".format (args.mode))

    net = models.FlowNet2(args)
    checkpoint = torch.load(args.weight_file)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    if args.mode == 'cuda':
        net.cuda()

    images, image_list = load_sequence(args.img1, args.img2)
    start_time = time.time()
    if args.mode == 'cuda':
        images = images.cuda()
    output = net(images)
    disps = output[0].data.cpu().numpy().transpose(1, 2, 0)
    end_time = time.time()
    print ("Inference took {:0.4f} seconds".format(end_time-start_time))
    # save the visualization of one disparity
    flow = flow_utils.flow2img (disps)
    plt.imshow(flow)
    plt.show()

