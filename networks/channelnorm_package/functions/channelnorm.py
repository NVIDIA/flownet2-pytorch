import torch
from torch.autograd import Function
from .._ext import channelnorm


class ChannelNormFunction(Function):

    def __init__(self, norm_deg=2):
        super(ChannelNormFunction, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        # self.save_for_backward(input1)

        assert(input1.is_contiguous() == True)

        with torch.cuda.device_of(input1):
            b, _, h, w = input1.size()
            output = input1.new().resize_(b, 1, h, w).zero_()

            channelnorm.ChannelNorm_cuda_forward(input1, output, self.norm_deg)
        self.save_for_backward(input1, output)

        return output

    def backward(self, gradOutput):
        input1, output = self.saved_tensors

        with torch.cuda.device_of(input1):
            b, c, h, w = input1.size()
            gradInput1 = input1.new().resize_(b,c,h,w).zero_()

            channelnorm.ChannelNorm_cuda_backward(input1, output, gradOutput, gradInput1, self.norm_deg)

        return gradInput1