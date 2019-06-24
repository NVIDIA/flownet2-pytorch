import os
import torch
import numpy as np
import argparse

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
from PIL import Image
from math import ceil

from pdb import set_trace

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    state_dict = torch.load("./FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(state_dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    img1_fn = "./flownet2-docker/data/0000000-imgL.png"
    img2_fn = "./flownet2-docker/data/0000001-imgL.png"
    pim1 = read_gen(img1_fn)
    pim2 = read_gen(img2_fn)
    # return numpy array with shape h,w,3
    
    img1 = Image.open(img1_fn)
    img2 = Image.open(img2_fn)
    assert(img1.size == img2.size)
    width, height = img1.size
    divisor = 64.
    adapted_width = int(ceil(width/divisor) * divisor)
    adapted_height = int(ceil(height/divisor) * divisor)
    img1 = img1.resize((adapted_width,adapted_height),Image.BICUBIC)
    img2 = img1.resize((adapted_width,adapted_height),Image.BICUBIC)
    pim1 = np.array(img1)
    pim2 = np.array(img2)
    
    assert(pim1.shape == pim2.shape)
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()
    data = result.data.cpu().numpy().transpose(1, 2, 0)

    cmp_path =  "./flownet2-docker/flow.flo"
    if os.path.isfile( cmp_path):
        cmp_data = read_gen(cmp_path)
        # resize channels individually
        if width != adapted_width or height != adapted_height:
            flow_u = Image.fromarray(data[:,:,0]).resize((width, height))
            flow_v = Image.fromarray(data[:,:,1]).resize((width, height))
            data = np.stack((flow_u,flow_v),axis=2)
        print("Doing comparison: ", np.linalg.norm(data - cmp_data) )

    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()
    
    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("./flow.flo", data)
    print("wrote flow.flo")
