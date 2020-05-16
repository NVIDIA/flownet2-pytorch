"""
Copyright 2020 Samim Taray

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from torch.nn.modules.module import Module
from torch.autograd import Function, Variable


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.clone().detach()
        # naive loop implementation from original flownet works well
        image_data = input1
        warped_data = output
        
        # for x in range(w):
        #     for y in range(h):
                
        #         fx = input2[0, 0, y, x]
        #         fy = input2[0, 1, y, x]
                
        #         x2 = x + fx
        #         y2 = y + fy

        #         if x2>=0 and y2>=0 and x2< w and y2 < h:
                
        #             ix2_L = int(x2)
        #             iy2_T = int(y2)
        #             ix2_R = min(ix2_L+1, w-1)
        #             iy2_B = min(iy2_T+1, h-1)

        #             alpha=x2-ix2_L
        #             beta=y2-iy2_T

        #             for c in range(3):
        #                 TL = image_data[:, c, iy2_T, ix2_L]
        #                 TR = image_data[:, c, iy2_T, ix2_R]
        #                 BL = image_data[:, c, iy2_B, ix2_L]
        #                 BR = image_data[:, c, iy2_B, ix2_R]

        #                 warped_data[:, c, y, x] = \
        #                     (1-alpha)*(1-beta)*TL + \
        #                     alpha*(1-beta)*TR + \
        #                     (1-alpha)*beta*BL + \
        #                     alpha*beta*BR
                
        #         else:
        #             for c in range(3):
        #                 warped_data[:, c, y, x] = 0.0

        # Vectorized implementation
        for batch in range(b):
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
            x = x.to(input1.device)
            y = y.to(input1.device)
            fx = input2[batch, 0, y, x]
            fy = input2[batch, 1, y, x]
            
            x2 = x.float() + fx
            y2 = y.float() + fy

            ix2_L = x2.long()
            iy2_T = y2.long()
            ix2_L = torch.clamp(ix2_L, 0, w-1)
            iy2_T = torch.clamp(iy2_T, 0, h-1)
            ix2_R = torch.clamp(ix2_L + 1, 0,  w-1)
            iy2_B = torch.clamp(iy2_T + 1, 0, h-1)
            
            alpha = x2-ix2_L.float()
            beta = y2-iy2_T.float()
            # for c in range(3):
            #     TL = image_data[:, c, iy2_T, ix2_L]
            #     TR = image_data[:, c, iy2_T, ix2_R]
            #     BL = image_data[:, c, iy2_B, ix2_L]
            #     BR = image_data[:, c, iy2_B, ix2_R]

            #     warped_data[:, c, :, :] = \
            #         (1-alpha)*(1-beta)*TL + \
            #         alpha*(1-beta)*TR + \
            #         (1-alpha)*beta*BL + \
            #         alpha*beta*BR

            TL = image_data[batch, :, iy2_T, ix2_L]
            TR = image_data[batch, :, iy2_T, ix2_R]
            BL = image_data[batch, :, iy2_B, ix2_L]
            BR = image_data[batch, :, iy2_B, ix2_R]
            #Interpolation
            warped_data[batch, :, :, :] = (1-alpha) * (1-beta) * TL + \
                                    alpha * (1-beta) * TR + \
                                    (1-alpha) * beta * BL + \
                                    alpha * beta * BR

        
        return warped_data

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size)

        return grad_input1, grad_input2, None


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size)
