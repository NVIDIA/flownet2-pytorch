import torch
from torch.autograd import Function, Variable
from torch.nn.modules.module import Module
# import channelnorm_cpp

class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, c, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()
        output = torch.pow(input1, norm_deg)
        output = torch.sqrt( torch.sum(output, dim=1))
        ctx.save_for_backward(input1, output)
        ctx.norm_deg = norm_deg
        return output.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Not Implemented!
        """
        input1, output = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())

        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)

