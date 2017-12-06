#ifdef __cplusplus
    extern "C" {
#endif

void Resample2d_kernel_forward(THCState* state, THCudaTensor* input1, THCudaTensor* input2, THCudaTensor* output, int kernel_size);

void Resample2d_kernel_backward(THCState* state, THCudaTensor* input1, THCudaTensor* input2, THCudaTensor* gradOutput, THCudaTensor* gradInput1, THCudaTensor* gradInput2, int kernel_size);

#ifdef __cplusplus
    }
#endif