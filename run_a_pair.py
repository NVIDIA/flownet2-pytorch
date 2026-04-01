import torch
import numpy as np
import argparse

from models import FlowNet2
from utils.frame_utils import read_gen

class Args():
    fp16 = False
    rgb_max = 255.

def get_flow(img1, img2, weights):
    # initial a Net
    args = Args()
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load(weights)
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(img1)
    pim2 = read_gen(img2)
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()
    data = result.data.cpu().numpy().transpose(1, 2, 0)
    return data
