"""
Copyright 2020 Samim Taray

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
from torch.nn import ZeroPad2d
# import correlation_cuda
import code

class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def extractwindow(self, f2pad, i, j):
        hindex = torch.tensor( range(i,i+(2*self.max_displacement)+1, self.stride2) )
        windex = torch.tensor( range(j,j+(2*self.max_displacement)+1, self.stride2) )
        # Advanced indexing logic. Ref: https://github.com/pytorch/pytorch/issues/1080
        # the way advance indexing works: 
        # ---> f2pad[:, :, hindex] chose value at f2pad at hindex location, then 
        # ---> appending [:, :, :, windex] to it only choses values at windex.
        # ---> Thus it choses value at the alternative location of f2pad
        # win = f2pad[:,:, i:i+(2*self.max_displacement)+1, j:j+(2*self.max_displacement)+1]
        
        win = f2pad[:, :, hindex][:, :, :, windex]
        return win
        
    def forward(self, f1, f2):
        self.save_for_backward(f1, f2)
        f1b = f1.shape[0] #batch 
        f1c = f1.shape[1] #channel
        f1h = f1.shape[2] #height
        f1w = f1.shape[3] #width

        f2b = f2.shape[0] #batch
        f2c = f2.shape[1] #channel
        f2h = f2.shape[2] #height
        f2w = f2.shape[3] #width

        # generate padded f2 
        padder = ZeroPad2d(self.pad_size)
        f2pad = padder(f2)
        
        # Define output shape and initialize it
        outc = (2*(self.max_displacement/self.stride2)+1) * (2*(self.max_displacement/self.stride2)+1)
        outc = int(outc) # number of output channel
        outb = f1b       # size of output batch
        outh = f1h       # size of output height
        outw = f1w       # size of output width
        output = torch.ones((outb, outc, outh, outw))
        # this gives device type
        output = output.to(f1.device)
        
        for i in range(f1h):
            for j in range(f1w):
                # Extract window W around i,j from f2pad of size (1X256X21X21)
                win = self.extractwindow(f2pad, i, j)
                # Extract kernel: size [1, 256, 1, 1]
                k = f1[:, :, i, j].unsqueeze(2).unsqueeze(3)
                # boradcasting multiplication along channel dimension
                # it multiplies all the 256 element of k to win and keep the result as it is
                # size of mult: 1, 256, 21, 21
                mult = win * k
                # Sum along channel dimension to get dot product. size 1X21X21
                inner_prod = torch.sum(mult, dim = 1)
                
                # Flatten last 2 dimensions h,w to one dimension of h*w = no of channels in output
                # size 1X1X1X441
                inner_prod = inner_prod.flatten(-2, -1).unsqueeze(1).unsqueeze(1)                
                output[:, :, i, j] = inner_prod
        # return the average 
        return output/f1c

    def backward(self, grad_output):
        """
        Not Implemented!
        """
        input1, input2 = self.saved_tensors        
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return grad_input1, grad_input2


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)

        return result

