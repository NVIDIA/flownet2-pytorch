import torch
from torch.autograd import Function
from .._ext import resample2d


class Resample2dFunction(Function):

    def __init__(self, kernel_size=1):
        super(Resample2dFunction, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)

        assert(input1.is_contiguous() == True)
        assert(input2.is_contiguous() == True)

        with torch.cuda.device_of(input1):
            _, d, _, _ = input1.size() 
            b, _, h, w = input2.size()
            output = input1.new().resize_(b, d, h, w).zero_()

            resample2d.Resample2d_cuda_forward(input1, input2, output, self.kernel_size)

        return output

    def backward(self, gradOutput):
        input1, input2 = self.saved_tensors

        assert(gradOutput.is_contiguous() == True)
        
        with torch.cuda.device_of(input1):
            b, c, h, w = input1.size()
            gradInput1 = input1.new().resize_(b,c,h,w).zero_()

            b, c, h, w = input2.size()
            gradInput2 = input2.new().resize_(b,c,h,w).zero_()

            resample2d.Resample2d_cuda_backward(input1, input2, gradOutput, gradInput1, gradInput2, self.kernel_size)

        return gradInput1, gradInput2