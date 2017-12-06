#include <THC.h>
#include <THCGeneral.h>

#include "Resample2d_kernel.h"

extern THCState* state;

int Resample2d_cuda_forward(THCudaTensor* input1, THCudaTensor* input2, THCudaTensor* output, int kernel_size) {
    Resample2d_kernel_forward(state, input1, input2, output, kernel_size);
    return 1;
}

int Resample2d_cuda_backward(THCudaTensor* input1, THCudaTensor* input2, THCudaTensor* gradOutput, THCudaTensor* gradInput1, THCudaTensor* gradInput2, int kernel_size) {
    Resample2d_kernel_backward(state, input1, input2, gradOutput, gradInput1, gradInput2, kernel_size);

    return 1;
}