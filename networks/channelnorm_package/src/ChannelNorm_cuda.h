int ChannelNorm_cuda_forward(THCudaTensor* input1, THCudaTensor* output, int norm_deg);

int ChannelNorm_cuda_backward(THCudaTensor* input1, THCudaTensor* output, THCudaTensor* gradOutput, THCudaTensor* gradInput1, int norm_deg);