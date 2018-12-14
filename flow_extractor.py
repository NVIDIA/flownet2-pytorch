import argparse, os
import numpy as np
import glob

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
from utils import flow_utils, tools
from datasets import ImagesFromFolderInference
import cortex.utils

import logging
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    print('#'*60)

    ######################################################################################
    ######################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument("--rgb_max", type=float, default = 255.)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--image_extension', default='jpg', type=str, help='image extension (jpg | png)')
    parser.add_argument('--flow_output_type', default='jpg', type=str, help='format to save flow (jpg | flo)')
    parser.add_argument('--checkpoint', required=False, default='./pretrained/FlowNet2_checkpoint.pth.tar', type=str, help='path to latest checkpoint')
    parser.add_argument('--image_dir_in', required=True, type=str, help='directory containing images to extract flow')
    parser.add_argument('--flow_dir_out', required=True, type=str, help='directory destination for flow saving')
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

    ######################################################################################
    ######################################################################################

    args = parser.parse_args()
    args.model_class = tools.module_to_dict(models)[args.model]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.flow_output_type in ('jpg', 'flo')

    ######################################################################################
    ######################################################################################

    # Intialize the model
    logging.info('Initialized FlowNet2 model.')
    model = models.FlowNet2(args, batchNorm=False, div_flow=20.)
    if args.cuda:
        model = model.cuda()
    model.eval()

    ######################################################################################
    ######################################################################################

    # Restore weights from checkpoint
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('Restored checkpoint file from: {}'.format(args.checkpoint))
    else:
        raise FileNotFoundError('no existing checkpoint found...')

    ######################################################################################
    ######################################################################################

    num_image_files = len(glob.glob(os.path.join(args.image_dir_in, '*.{}'.format(args.image_extension))))
    if os.path.isdir(args.image_dir_in) and num_image_files == 0:
        # Process subdirectory mode
        image_subdirs = cortex.utils.find_subdirs(args.image_dir_in)
    else:
        # Just process all the JPG images in this folder
        image_subdirs = [args.image_dir_in]

    ######################################################################################
    ######################################################################################

    # Create main output dir
    if not os.path.exists(args.flow_dir_out):
        os.makedirs(args.flow_dir_out)
        logging.info('Creating output directory: {}'.format(args.flow_dir_out))

    ######################################################################################
    ######################################################################################

    for subdir_idx, image_subdir in enumerate(image_subdirs):

        num_image_files = len(glob.glob(os.path.join(image_subdir, '*.{}'.format(args.image_extension))))

        if num_image_files == 0:
            logging.info('skipping directory because no image files found: {}'.format(image_subdir))
            continue

        # Intialize the data loader for this video
        if os.path.exists(args.image_dir_in):
            inference_size = [-1, -1]  # largest possible
            dataset = ImagesFromFolderInference(image_subdir, inference_size, extension=args.image_extension)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True)
            logging.info('Succesfully initialized data set with {} images.'.format(len(dataset)))
        else:
            raise NotADirectoryError('Input image folder does not exist: {}'.format(args.image_dir_in))

        ######################################################################################
        ######################################################################################

        # Create output directory
        subdir_name = os.path.basename(image_subdir)
        flow_subdir_out = os.path.join(args.flow_dir_out, subdir_name)
        if not os.path.exists(flow_subdir_out):
            os.makedirs(flow_subdir_out)

        ######################################################################################
        ######################################################################################

        num_batches = int(np.ceil(len(dataset) / args.batch_size))

        # Array for storing all the minimum/maximum values
        minmax_arr = []

        for batch_idx, (data, target) in enumerate(data_loader):

            logging.info('video {}/{} with name \'{}\' | processing batch {}/{}...'.format(subdir_idx+1, len(image_subdirs), subdir_name, batch_idx+1, num_batches))

            # Prepare inputs for forward pass
            if args.cuda:
                data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]

            # Actual forward pass through the network
            with torch.no_grad():
                output = model(data[0])

            # Saving the outputs
            for example_idx in range(output.shape[0]):

                flow_single = output[example_idx].data.cpu().numpy().transpose(1, 2, 0)
                output_file = os.path.join(flow_subdir_out, '{:06d}'.format(batch_idx*args.batch_size + example_idx))

                if args.flow_output_type == 'flo':
                    output_file += '.flo'
                    flow_utils.writeFlow(output_file, flow_single)
                else:
                    output_file += '.jpg'
                    minmax_curr = flow_utils.writeFlowJPEG(output_file, flow_single)
                    minmax_arr.append(minmax_curr)

            # After every batch, save all the min/max values to disk
            minmax_file = os.path.join(flow_subdir_out, "minmax_values.npy")
            np.save(minmax_file, np.asarray(minmax_arr, np.float32))

        print('#'*60)