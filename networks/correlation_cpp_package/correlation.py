import torch
import correlation

class CorrelationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f1, f2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2):
        output = correlation.forward(f1, f2, pad_size, kernel_size, max_displacement, stride1, stride2)
        # ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx):
        """
        Not Implemented!
        """
        output = None
    
class Correlation(torch.nn.Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2)
        return result
