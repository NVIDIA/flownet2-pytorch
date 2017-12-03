import torch
from torch.autograd import Function
from .._ext import correlation


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply
        # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)

        assert(input1.is_contiguous() == True)
        assert(input2.is_contiguous() == True)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation.Correlation_forward_cuda(input1, input2, rbot1, rbot2, output, 
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors
        
        assert(grad_output.is_contiguous() == True)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation.Correlation_backward_cuda(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return grad_input1, grad_input2