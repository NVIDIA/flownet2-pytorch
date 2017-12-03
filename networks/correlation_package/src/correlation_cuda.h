int Correlation_forward_cuda(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *rInput1, THCudaTensor *rInput2, 
                       THCudaTensor *output, 
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply);

int Correlation_backward_cuda(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *rInput1, THCudaTensor *rInput2, 
                       THCudaTensor *gradOutput, THCudaTensor *gradInput1, THCudaTensor *gradInput2, 
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply);

