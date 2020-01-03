#!/usr/bin/env python
import argparse
import os
import shutil
import tempfile

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import flow_utils, tools
from models import FlowNet2
from datasets import ImagesFromFolder, ImagesFromSubFolders


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'manifest_folder',
        help="path to a dataset manifest folder, at minimum, must contain a "
             "folder named data, subfolders of which contain continous sets "
             "of images.")
    inpopt = parser.add_argument_group('input options')
    outopt = parser.add_argument_group('output options')
    runopt = parser.add_argument_group('run options')
    spdopt = parser.add_argument_group('speedup options')
    inpopt.add_argument(
        '--iext', default='jpg', choices=['jpg', 'jpeg', 'png'],
        help="file extension of input images within subfolders"
    )
    runopt.add_argument(
        '--mode', '-m', default='resample', choices=['resample', 'crop'],
        help="method for reshaping input frames to legal model input shape")
    outopt.add_argument(
        '--suffix', default='_feature-flow',
        help="suffix to append to input file basenames")
    outopt.add_argument(
        '--resize-outputs', action='store_true', #@FIXME implement this
        help="NOT IMPLEMENTED reshapes outputs to match input dimensions")
    spdopt.add_argument(
        '--batch-size', '-b', type=int, default=64,
        help="batch size to use during inference, default is 64")
    spdopt.add_argument(
        '--ncpus', default=os.cpu_count(),
        help="number of cpus to use for loading data, default is all visible")
    spdopt.add_argument(
        '--ngpus', default=torch.cuda.device_count(),
        help="number of gpus to use for loading data, default is all visible")
    runopt.add_argument(
        '--weights-path', default='checkpoints/FlowNet2_checkpoint.pth.tar',
        help="path to model weights, default is to use Flownet2 weights"
    )

    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = args.ngpus
    run_inference(**vars(args))


def run_inference(manifest_folder, batch_size=16, ncpus=1, ngpus=1,
                  resize_outputs=False, mode='resample',
                  suffix='_feature-flow', batch_norm=False, div_flow=20.,
                  rgb_max=1., iext='jpg',
                  weights_path='checkpoints/FlowNet2_checkpoint.pth.tar'):
    model = FlowNet2(
        batchNorm=batch_norm,
        div_flow=div_flow,
        rgb_max=rgb_max,
        fp16=False,
    )
    if weights_path:
        model.load_state_dict(torch.load(weights_path)["state_dict"])
    if ngpus:
        model = model.cuda()
        if ngpus > 1:
            model = torch.nn.parallel.DataParallel(model, list(range(ngpus)))
    model.eval()

    data_folder = os.path.join(manifest_folder, 'data')

    feature_folder = os.path.join(
        os.path.dirname(data_folder), 'feature'
    )
    suffix = '_feature-flow'
    def get_output_name(input_name):
        input_folder = os.path.basename(os.path.dirname(input_name))
        output_folder = os.path.join(
            feature_folder, '%s%s' % (input_folder, suffix))
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder,
            '%s%s' % (os.path.splitext(os.path.basename(input_name))[0], suffix + '.flo'))
        return output_file
    # create dataset
    dataset = ImagesFromSubFolders(data_folder, iext=iext)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=ncpus)
    progress = tqdm(tools.IteratorTimer(data_loader),
                    ncols=100, total=len(data_loader), leave=True, desc='batch progress')

    # run model on dataset
    with torch.no_grad():
        for batch_idx, (batch, filenames) in enumerate(progress):
            out_files = [get_output_name(f) for f in filenames]
            if all([os.path.exists(f) for f in out_files]):
                print('all outputs for batch %s already exist, skipping...' % batch_idx)
                continue
            if ngpus:
                batch = [b.cuda(async=True) for b in batch]
            batch = batch[0]
            out_flow = model(batch)
            for i in range(out_flow.shape[0]):
                _pflow = out_flow[i].data.cpu().numpy().transpose(1, 2, 0)  # u, v last
                out_file = out_files[i]
                if not os.path.exists(out_file):
                    flow_utils.writeFlow(out_file, _pflow)
                else:
                    print('out file already exists: %s' % out_file)


if __name__ == '__main__':
    cli()
