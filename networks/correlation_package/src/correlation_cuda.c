#include <THC/THC.h>
#include <stdio.h>

#include "correlation_cuda_kernel.h"

#define real float

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

int Correlation_forward_cuda(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *rInput1, THCudaTensor *rInput2, THCudaTensor *output,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{

  int batchSize = input1->size[0];
  int nInputChannels = input1->size[1];
  int inputHeight = input1->size[2];
  int inputWidth = input1->size[3];

  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;

  int paddedInputHeight = inputHeight + 2 * pad_size;
  int paddedInputWidth = inputWidth + 2 * pad_size;

  int nOutputChannels = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1);

  int outputHeight = ceil((float)(paddedInputHeight - 2 * border_radius) / (float)stride1);
  int outputwidth = ceil((float)(paddedInputWidth - 2 * border_radius) / (float)stride1);

  THCudaTensor_resize4d(state, rInput1, batchSize, paddedInputHeight, paddedInputWidth, nInputChannels);
  THCudaTensor_resize4d(state, rInput2, batchSize, paddedInputHeight, paddedInputWidth, nInputChannels);
  THCudaTensor_resize4d(state, output, batchSize, nOutputChannels, outputHeight, outputwidth);

  THCudaTensor_fill(state, rInput1, 0);
  THCudaTensor_fill(state, rInput2, 0);
  THCudaTensor_fill(state, output, 0);

  int success = 0;
  success = Correlation_forward_cuda_kernel(  THCudaTensor_data(state, output),
                                              THCudaTensor_size(state, output, 0),
                                              THCudaTensor_size(state, output, 1),
                                              THCudaTensor_size(state, output, 2),
                                              THCudaTensor_size(state, output, 3),
                                              THCudaTensor_stride(state, output, 0),
                                              THCudaTensor_stride(state, output, 1),
                                              THCudaTensor_stride(state, output, 2),
                                              THCudaTensor_stride(state, output, 3),

                                              THCudaTensor_data(state, input1),
                                              THCudaTensor_size(state, input1, 1),
                                              THCudaTensor_size(state, input1, 2),
                                              THCudaTensor_size(state, input1, 3),
                                              THCudaTensor_stride(state, input1, 0),
                                              THCudaTensor_stride(state, input1, 1),
                                              THCudaTensor_stride(state, input1, 2),
                                              THCudaTensor_stride(state, input1, 3),

                                              THCudaTensor_data(state, input2),
                                              THCudaTensor_size(state, input2, 1),
                                              THCudaTensor_stride(state, input2, 0),
                                              THCudaTensor_stride(state, input2, 1),
                                              THCudaTensor_stride(state, input2, 2),
                                              THCudaTensor_stride(state, input2, 3),

                                              THCudaTensor_data(state, rInput1),
                                              THCudaTensor_data(state, rInput2),
                                              
                                              pad_size,
                                              kernel_size,
                                              max_displacement,
                                              stride1,
                                              stride2,
                                              corr_type_multiply,

                                              THCState_getCurrentStream(state));

    THCudaTensor_free(state, rInput1);
    THCudaTensor_free(state, rInput2);

  //check for errors
  if (!success) {
    THError("aborting");
  }

  return 1;

}

int Correlation_backward_cuda(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *rInput1, THCudaTensor *rInput2, THCudaTensor *gradOutput, 
                       THCudaTensor *gradInput1, THCudaTensor *gradInput2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{

  int batchSize = input1->size[0];
  int nInputChannels = input1->size[1];
  int paddedInputHeight = input1->size[2]+ 2 * pad_size;
  int paddedInputWidth = input1->size[3]+ 2 * pad_size;

  int height = input1->size[2];
  int width = input1->size[3];

  THCudaTensor_resize4d(state, rInput1, batchSize, paddedInputHeight, paddedInputWidth, nInputChannels);
  THCudaTensor_resize4d(state, rInput2, batchSize, paddedInputHeight, paddedInputWidth, nInputChannels);
  THCudaTensor_resize4d(state, gradInput1, batchSize, nInputChannels, height, width);
  THCudaTensor_resize4d(state, gradInput2, batchSize, nInputChannels, height, width);
  
  THCudaTensor_fill(state, rInput1, 0);
  THCudaTensor_fill(state, rInput2, 0);
  THCudaTensor_fill(state, gradInput1, 0);
  THCudaTensor_fill(state, gradInput2, 0);

  int success = 0;
  success = Correlation_backward_cuda_kernel(
                                              THCudaTensor_data(state, gradOutput),
                                              THCudaTensor_size(state, gradOutput, 0),
                                              THCudaTensor_size(state, gradOutput, 1),
                                              THCudaTensor_size(state, gradOutput, 2),
                                              THCudaTensor_size(state, gradOutput, 3),
                                              THCudaTensor_stride(state, gradOutput, 0),
                                              THCudaTensor_stride(state, gradOutput, 1),
                                              THCudaTensor_stride(state, gradOutput, 2),
                                              THCudaTensor_stride(state, gradOutput, 3),

                                              THCudaTensor_data(state, input1),
                                              THCudaTensor_size(state, input1, 1),
                                              THCudaTensor_size(state, input1, 2),
                                              THCudaTensor_size(state, input1, 3),
                                              THCudaTensor_stride(state, input1, 0),
                                              THCudaTensor_stride(state, input1, 1),
                                              THCudaTensor_stride(state, input1, 2),
                                              THCudaTensor_stride(state, input1, 3),

                                              THCudaTensor_data(state, input2),
                                              THCudaTensor_stride(state, input2, 0),
                                              THCudaTensor_stride(state, input2, 1),
                                              THCudaTensor_stride(state, input2, 2),
                                              THCudaTensor_stride(state, input2, 3),

                                              THCudaTensor_data(state, gradInput1),
                                              THCudaTensor_stride(state, gradInput1, 0),
                                              THCudaTensor_stride(state, gradInput1, 1),
                                              THCudaTensor_stride(state, gradInput1, 2),
                                              THCudaTensor_stride(state, gradInput1, 3),

                                              THCudaTensor_data(state, gradInput2),
                                              THCudaTensor_size(state, gradInput2, 1),
                                              THCudaTensor_stride(state, gradInput2, 0),
                                              THCudaTensor_stride(state, gradInput2, 1),
                                              THCudaTensor_stride(state, gradInput2, 2),
                                              THCudaTensor_stride(state, gradInput2, 3),

                                              THCudaTensor_data(state, rInput1),
                                              THCudaTensor_data(state, rInput2),
                                              pad_size,
                                              kernel_size,
                                              max_displacement,
                                              stride1,
                                              stride2,
                                              corr_type_multiply,
                                              THCState_getCurrentStream(state));

    THCudaTensor_free(state, rInput1);
    THCudaTensor_free(state, rInput2);

  if (!success) {
    THError("aborting");
  }
  return 1;
}
