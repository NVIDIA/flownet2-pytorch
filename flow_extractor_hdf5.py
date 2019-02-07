# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2019-02-06

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse
import numpy as np
import glob
import shutil
import tqdm

import h5py
import jpeg4py as jpeg

import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
from utils import flow_utils, tools
from datasets import ImagesFromFolderInference

import cortex.utils
import cortex.vision.flow

################################################################################

def extract_frames_from_hdf5_dir(directory, tmp_directory, dset_prefix='video_'):
    '''
    Extracts frames as JPG from all HDF5 containers in a specified directory.
    
    Args:
        directory (str): Input directory containing HDF5 files
        tmp_directory (str): Where to save temporary image files
        dset_prefix (str, optional): Defaults to 'video_'. Name of frames dataset.
    
    Returns:
        list: List of examples to process by flow extractor
    '''
    if not os.path.isdir(directory):
        raise FileNotFoundError('HDF5 directory does not exist: {}'.format(directory))
    cortex.utils.mkdirs(tmp_directory)
    hdf5_files = cortex.utils.find_files(directory, '.h5')
    if not hdf5_files:
        raise FileNotFoundError('Directory contains no HDF5 files: {}'.format(directory))
    example_list = []
    for hdf5_file in hdf5_files:  # <== TODO: fixme
        tmp_subdirectory = os.path.join(tmp_directory, cortex.utils.basename(hdf5_file))
        cortex.utils.mkdirs(tmp_directory)
        example_list += extract_frames_from_hdf5(hdf5_file, tmp_subdirectory, dset_prefix)
    return example_list

def extract_frames_from_hdf5(hdf5_file, tmp_directory, dset_prefix='video_'):
    '''
    Extracts all frames as JPG from a specified HDF5 container
    
    Args:
        hdf5_file (str): Input HDF5 file containing 
        tmp_directory (str): Where to save temporary image files
        dset_prefix (str, optional): Defaults to 'video_'. Name of frames dataset.
    
    Returns:
        list: List of examples to process by flow extractor
    '''
    if not os.path.isfile(hdf5_file):
        raise FileNotFoundError('HDF5 file does not exist: {}'.format(hdf5_file))
    cortex.utils.mkdirs(tmp_directory)
    with h5py.File(hdf5_file) as hf:
        example_list = []
        frame_containers = list(filter(lambda x: x.startswith(dset_prefix), list(hf.keys())))
        num_videos = len(frame_containers)
        print('Extracting all frames from {} examples inside HDF5 file: {}'.format(num_videos, hdf5_file))
        for example_idx, dset_name in enumerate(tqdm.tqdm(frame_containers)):

            # Check if flow container already exists
            dset_index = int(re.findall(r'\d+', dset_name)[-1])
            if "flow_{:06d}".format(dset_index) in hf and "flow_minmax_{:06d}".format(dset_index) in hf:
                print('Skipping because flow already exists in HDF5 container...')
                continue
            
            video_subdir = os.path.join(tmp_directory, dset_name)
            cortex.utils.mkdirs(video_subdir)            
            for frame_idx, frame_raw in enumerate(hf[dset_name]):
                frame_jpg_encode = np.fromstring(frame_raw, dtype=np.uint8)
                frame_decode = jpeg.JPEG(frame_jpg_encode).decode(pixfmt=jpeg.TJPF_BGR) # BGR
                tmp_output_file = os.path.join(video_subdir, '{:06}.jpg'.format(frame_idx))
                cv2.imwrite(tmp_output_file, frame_decode)
            example_list.append({'hdf5_file': hdf5_file, 'dset_name': dset_name, 'tmp_image_dir': video_subdir})
    return example_list

################################################################################

def extract_flow_to_hdf5(model, example_list, cleanup_tmp_dirs=True):

    for example_idx, example in enumerate(example_list):
        
        # Parse dataset number from string 'video_000010' ==> int(10)
        dset_index = int(re.findall(r'\d+', example['dset_name'])[-1])
        
        print('Working on example {}/{}...'.format(example_idx+1, len(example_list)))
        tmp_image_dir = example['tmp_image_dir']
        num_image_files = len(cortex.utils.find_files(tmp_image_dir, 'jpg'))
        if num_image_files == 0:
            print('Skipping directory because no image files found: {}'.format(tmp_image_dir))
            continue

        # Intialize the data loader for this video
        inference_size = [-1, -1]  # largest possible
        dataset = ImagesFromFolderInference(tmp_image_dir, inference_size, extension='jpg')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True)
        print('Succesfully initialized dataloader with {} image pairs.'.format(len(dataset)))

        ######################################################################################
        ######################################################################################

        flow_minmax = []
        flow_images = []

        num_batches = int(np.ceil(len(dataset) / args.batch_size))

        for batch_idx, (data, target) in enumerate(data_loader):

            # Prepare inputs for forward pass
            if args.cuda:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]

            # Actual forward pass through the network
            with torch.no_grad():
                # Shape = [N,2,H,W]
                output = model(data[0])

            # Saving the outputs
            for example_idx in range(output.shape[0]):
                flow_single = output[example_idx].data.cpu().numpy().transpose(1, 2, 0)
                # Normalize and get 3-channel image
                flow_u_norm, min_u, max_u = cortex.vision.flow.normalize_flow(flow_single[:,:,0])
                flow_v_norm, min_v, max_v = cortex.vision.flow.normalize_flow(flow_single[:,:,1])
                flow_as_jpg = np.dstack((flow_u_norm, flow_v_norm, np.zeros_like(flow_u_norm)))
                flow_as_jpg = (flow_as_jpg*255.0).astype(np.uint8)
                # Save results
                flow_minmax.append((min_u, max_u, min_v, max_v))
                flow_images.append(flow_as_jpg)

        # All batches are done, now write flow frames to HDF5 file
        with h5py.File(example['hdf5_file'], 'a') as hf:
            print('Writing results to HDF5 file...')
            # Create dataset for frames
            dt_frames  = h5py.special_dtype(vlen=np.dtype('uint8'))
            dset_flow = hf.create_dataset("flow_{:06d}".format(dset_index), shape=(len(flow_images),), dtype=dt_frames)
            for frame_idx, flow_image in enumerate(flow_images):
                # Apply JPG compression to the raw video frame
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                frame_jpg_encode = cv2.imencode(".jpg", flow_image, encode_params)
                frame_jpg_encode = frame_jpg_encode[1].tostring()
                dset_flow[frame_idx] = np.fromstring(frame_jpg_encode, dtype='uint8')
            flow_minmax = np.asarray(flow_minmax, np.float64)
            dset_flow = hf.create_dataset("flow_minmax_{:06d}".format(dset_index), data=flow_minmax)

        if cleanup_tmp_dirs:
            print('cleaning up temporary image directory: {}'.format(tmp_image_dir))
            shutil.rmtree(tmp_image_dir)
        
        print('#'*60)
    print('Done.')


################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument("--rgb_max", type=float, default = 255.)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--checkpoint', required=False, default='./pretrained/FlowNet2_checkpoint.pth.tar', type=str, help='path to latest checkpoint')
    parser.add_argument('--hdf5_input_path', required=True, type=str, help='path to HDF5 file containing images to extract flow')
    parser.add_argument('--hdf5_frames_dset', required=False, default='frames_', type=str, help='name of HDF5 dataset storing the video frames')
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

    ######################################################################################
    ######################################################################################

    args = parser.parse_args()
    args.model_class = tools.module_to_dict(models)[args.model]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    ################################################################################
    ################################################################################
    # Extract all the images into temp directory

    tmp_image_dir = '/tmp/flownet2/'
    cortex.utils.mkdirs(tmp_image_dir)

    if os.path.isfile(args.hdf5_input_path):
        # Only process a single HDF5 file
        print('Processing single HDF5 file: {}'.format(args.hdf5_input_path))
        example_list = extract_frames_from_hdf5(args.hdf5_input_path, tmp_image_dir)
    elif os.path.isdir(args.hdf5_input_path):
        # Process a directory containing multiple HDF5 files
        print('Processing multiple HDF5 files in directory: {}'.format(args.hdf5_input_path))
        example_list = extract_frames_from_hdf5_dir(args.hdf5_input_path, tmp_image_dir)
    else:
        raise FileNotFoundError('Specified path of HDF5 inputs could not be found: {}'.format(args.hdf5_input_path))

    if not example_list:
        print('No examples to process...Exiting.')
        exit(0)

    ######################################################################################
    ######################################################################################

    # Intialize the model
    print('Initialized FlowNet2 model.')
    model = models.FlowNet2(args, batchNorm=False, div_flow=20.)
    if args.cuda:
        model = model.cuda()
    model.eval()

    # Restore weights from checkpoint
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print('Restored checkpoint file from: {}'.format(args.checkpoint))
    else:
        raise FileNotFoundError('no existing checkpoint found...')

    ######################################################################################
    ######################################################################################

    extract_flow_to_hdf5(model, example_list)

    # Cleanup temp directory containing JPG frames
    shutil.rmtree(tmp_image_dir)
    
