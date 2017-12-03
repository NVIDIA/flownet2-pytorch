#ifdef __cplusplus
extern "C" {
#endif

    int Correlation_forward_cuda_kernel(/*THCudaTensor_data(state, output)*/ float *output,
        /*THCudaTensor_size(state, output, 0)*/ int ob,
        /*THCudaTensor_size(state, output, 1)*/ int oc,
        /*THCudaTensor_size(state, output, 2)*/ int oh,
        /*THCudaTensor_size(state, output, 3)*/ int ow,
        /*THCudaTensor_stride(state, output, 0)*/ int osb,
        /*THCudaTensor_stride(state, output, 1)*/ int osc,
        /*THCudaTensor_stride(state, output, 2)*/ int osh,
        /*THCudaTensor_stride(state, output, 3)*/ int osw,

        /*THCudaTensor_data(state, input1)*/ float *input1,
        /*THCudaTensor_size(state, input1, 1)*/ int ic,
        /*THCudaTensor_size(state, input1, 2)*/ int ih,
        /*THCudaTensor_size(state, input1, 3)*/ int iw,
        /*THCudaTensor_stride(state, input1, 0)*/ int isb,
        /*THCudaTensor_stride(state, input1, 1)*/ int isc,
        /*THCudaTensor_stride(state, input1, 2)*/ int ish,
        /*THCudaTensor_stride(state, input1, 3)*/ int isw,

        /*THCudaTensor_data(state, input2)*/ float *input2,
        /*THCudaTensor_size(state, input2, 1)*/ int gc,
        /*THCudaTensor_stride(state, input2, 0)*/ int gsb,
        /*THCudaTensor_stride(state, input2, 1)*/ int gsc,
        /*THCudaTensor_stride(state, input2, 2)*/ int gsh,
        /*THCudaTensor_stride(state, input2, 3)*/ int gsw,

        /*THCudaTensor_data(state, rInput1)*/ float *rInput1,
        /*THCudaTensor_data(state, rInput2)*/ float *rInput2,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2,
        int corr_type_multiply,
        /*THCState_getCurrentStream(state)*/ cudaStream_t stream);

    int Correlation_backward_cuda_kernel(   
        /*THCudaTensor_data(state, gradOutput)*/    float *gradOutput,
        /*THCudaTensor_size(state, gradOutput, 0)*/ int gob,
        /*THCudaTensor_size(state, gradOutput, 1)*/ int goc,
        /*THCudaTensor_size(state, gradOutput, 2)*/ int goh,
        /*THCudaTensor_size(state, gradOutput, 3)*/ int gow,
        /*THCudaTensor_stride(state, gradOutput, 0)*/ int gosb,
        /*THCudaTensor_stride(state, gradOutput, 1)*/ int gosc,
        /*THCudaTensor_stride(state, gradOutput, 2)*/ int gosh,
        /*THCudaTensor_stride(state, gradOutput, 3)*/ int gosw,

        /*THCudaTensor_data(state, input1)*/        float* input1,
        /*THCudaTensor_size(state, input1, 1)*/     int ic,
        /*THCudaTensor_size(state, input1, 2)*/     int ih,
        /*THCudaTensor_size(state, input1, 3)*/     int iw,
        /*THCudaTensor_stride(state, input1, 0)*/   int isb,
        /*THCudaTensor_stride(state, input1, 1)*/   int isc,
        /*THCudaTensor_stride(state, input1, 2)*/   int ish,
        /*THCudaTensor_stride(state, input1, 3)*/   int isw,

        /*THCudaTensor_data(state, input2)*/        float *input2,
        /*THCudaTensor_stride(state, input2, 0)*/   int gsb,
        /*THCudaTensor_stride(state, input2, 1)*/   int gsc,
        /*THCudaTensor_stride(state, input2, 2)*/   int gsh,
        /*THCudaTensor_stride(state, input2, 3)*/   int gsw,

        /*THCudaTensor_data(state, gradInput1)*/    float *gradInput1, 
        /*THCudaTensor_stride(state, gradInput1, 0)*/ int gisb,
        /*THCudaTensor_stride(state, gradInput1, 1)*/ int gisc,
        /*THCudaTensor_stride(state, gradInput1, 2)*/ int gish,
        /*THCudaTensor_stride(state, gradInput1, 3)*/ int gisw,

        /*THCudaTensor_data(state, gradInput2)*/      float *gradInput2,
        /*THCudaTensor_size(state, gradInput2, 1)*/   int ggc,
        /*THCudaTensor_stride(state, gradInput2, 0)*/ int ggsb,
        /*THCudaTensor_stride(state, gradInput2, 1)*/ int ggsc,
        /*THCudaTensor_stride(state, gradInput2, 2)*/ int ggsh,
        /*THCudaTensor_stride(state, gradInput2, 3)*/ int ggsw,

        /*THCudaTensor_data(state, rInput1)*/             float *rInput1,
        /*THCudaTensor_data(state, rInput2)*/             float *rInput2,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2,
        int corr_type_multiply,
        /*THCState_getCurrentStream(state)*/cudaStream_t stream);

#ifdef __cplusplus
}
#endif
