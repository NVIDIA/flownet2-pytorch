#!/usr/bin/env python
import argparse
import os
import shutil
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import FlowNet2
from datasets import ImagesFromFolder
from utils import flow_utils


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'manifest_folder',
        help="path to a dataset manifest folder")
    parser.add_argument(
        '--mode', '-m', default='resample', choices=['resample', 'crop'],
        help="method for reshaping input frames to legal model input shape")
    parser.add_argument(
        '--resize-outputs', action='store_true', #@FIXME implement this
        help="reshapes outputs to match input dimensions")
    parser.add_argument(
        '--batch-size', '-b', type=int)
    parser.add_argument(
        '--ncpus', default=os.cpu_count())
    parser.add_argument(
        '--ngpus', default=torch.cuda.device_count())

    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = args.ngpus
    main_wrapper(**vars(args))


def main_wrapper(manifest_folder, batch_size, ncpus=1, ngpus=1,
                 resize_outputs=False, mode='resample', suffix='_feature-flow', 
                 batch_norm=False, div_flow=20., rgb_max=255., iext='jpg'):
    model = FlowNet2(
        batchNorm=batch_norm,
        div_flow=div_flow,
        rgb_max=rgb_max,
        fp16=False,
    )
    if ngpus:
        model = model.cuda()
        if ngpus > 1:
            model = torch.nn.parallel.DataParallel(model, list(range(ngpus)))
    model.eval()

    data_folder = os.path.join(manifest_folder, 'data')
    for image_folder in os.listdir(data_folder):
        image_folder = os.path.join(data_folder, image_folder)
        inference(model, image_folder, ncpus)


def inference(model, input_folder, batch_size, ncpus, iext='jpg'):
    input_foldername = os.path.basename(input_folder)
    feature_folder = os.path.join(
        os.path.dirname(os.path.dirname(input_folder)), 'feature'
    )
    suffix = '_feature-flow'
    output_folder = os.path.join(
        feature_folder, '%s%s' % (input_foldername, suffix))
    os.makedirs(output_folder, exist_ok=True)
    # create dataset
    dataset = ImagesFromFolder(input_folder, iext=iext)
    first_frames = [x[0] for x in dataset.image_list]
    outputs = list(map(lambda x: x + suffix + '.flo', map(os.path.basename, first_frames)))
    final_outputs = set(os.listdir(output_folder))
    if set(outputs) <= final_outputs:
        print('flow files already exist for %s, skipping...' % output_folder)
        return
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=ncpus)
    # run model on dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        for batch_idx, (batch, _) in enumerate(data_loader):
            out_flow = model(batch)
            for i in range(out_flow.shape[0]):
                _pflow = out_flow[i].data.cpu().numpy().transpose(1, 2, 0)  # u, v last
                flow_index = batch_idx * batch_size + i
                flow_utils.writeFlow(os.path.join(temp_dir, outputs[flow_index]), _pflow)
        assert len(os.listdir(temp_dir)) == len(first_frames), "unequal number of inputs/outputs"
        shutil.rmtree(output_folder)
        shutil.move(temp_dir, output_folder)


if __name__ == '__main__':
    cli()
