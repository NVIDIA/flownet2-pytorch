int Resample2d_cuda_forward(THCudaTensor* input1, THCudaTensor* input2, THCudaTensor* output, int kernel_size);

int Resample2d_cuda_backward(THCudaTensor* input1, THCudaTensor* input2, THCudaTensor* gradOutput, THCudaTensor* gradInput1, THCudaTensor* gradInput2, int kernel_size);