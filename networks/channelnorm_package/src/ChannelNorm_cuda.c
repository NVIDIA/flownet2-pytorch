#include <THC.h>
#include <THCGeneral.h>

#include "ChannelNorm_kernel.h"

extern THCState* state;

int ChannelNorm_cuda_forward(THCudaTensor* input1, THCudaTensor* output, int norm_deg) {
    ChannelNorm_kernel_forward(state, input1, output, norm_deg);
    return 1;
}


int ChannelNorm_cuda_backward(THCudaTensor* input1, THCudaTensor* output, THCudaTensor* gradOutput, THCudaTensor* gradInput1, int norm_deg) {
    ChannelNorm_kernel_backward(state, input1, output, gradOutput, gradInput1, norm_deg);
    return 1;
}