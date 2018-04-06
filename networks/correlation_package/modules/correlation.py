from torch.nn.modules.module import Module

from ..functions.correlation import CorrelationFunction


class Correlation(Module):

    def __init__(self,
                 pad_size=0,
                 kernel_size=0,
                 max_displacement=0,
                 stride1=1,
                 stride2=2,
                 corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return CorrelationFunction.apply(input1, input2, self.pad_size,
                                         self.kernel_size,
                                         self.max_displacement, self.stride1,
                                         self.stride2, self.corr_multiply)
