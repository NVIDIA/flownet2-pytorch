import os
from glob import glob

import torch
import torch.utils.data as data
import torchvision.transforms as tvt
from PIL import Image

import utils.frame_utils as frame_utils


class ImagesFromFolder(data.Dataset):
    def __init__(self, root, iext='jpg', resample=True, crop=False, match_original_size=False):
        images = sorted( glob( os.path.join(root, '*.' + iext) ) )
        self.image_list = []
        for i in range(len(images)-1):
            im1 = images[i]
            im2 = images[i+1]
            self.image_list += [ [ im1, im2 ] ]

        self.size = len(self.image_list)
        assert self.size, "attempted to read an empty folder for flow at %s"  % root
        print('setting up %s' % root)
        print('number of image pairs found: %s' % self.size)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape
        self.input_size = (256, 256)  # could make this more flexible

        self.transforms = tvt.Compose(
            [tvt.Resize(self.input_size), # or center crop
             tvt.ToTensor()]
        )

    def __getitem__(self, index):
        img0 = Image.open(self.image_list[index][0])
        img1 = Image.open(self.image_list[index][1])
        images = torch.stack(list(map(self.transforms, [img0, img1])), dim=0)

        return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

    def __len__(self):
        return self.size

'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''
