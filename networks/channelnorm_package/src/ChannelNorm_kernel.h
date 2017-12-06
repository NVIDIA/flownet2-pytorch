#ifdef __cplusplus
    extern "C" {
#endif

void ChannelNorm_kernel_forward(THCState* state, THCudaTensor* input1, THCudaTensor* output, int norm_deg);


void ChannelNorm_kernel_backward(THCState* state, THCudaTensor* input1, THCudaTensor* output, THCudaTensor* gradOutput, THCudaTensor* gradInput1, int norm_deg);

#ifdef __cplusplus
    }
#endif