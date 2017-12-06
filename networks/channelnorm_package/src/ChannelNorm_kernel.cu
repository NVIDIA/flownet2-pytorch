#include <THC.h>
#include <THCGeneral.h>

#define CUDA_NUM_THREADS 512 
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])


#ifdef __cplusplus
    extern "C" {
#endif

__global__ void kernel_ChannelNorm_updateOutput(const int n, const float* input1, const long4 input1_size, const long4 input1_stride, float* output, const long4 output_size, const long4 output_stride, int norm_deg) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int dim_b = DIM0(output_size);
    int dim_c = DIM1(output_size);
    int dim_h = DIM2(output_size);
    int dim_w = DIM3(output_size);
    int dim_chw = dim_c * dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    int i1dim_c = DIM1(input1_size);
    int i1dim_h = DIM2(input1_size);
    int i1dim_w = DIM3(input1_size);
    int i1dim_chw = i1dim_c * i1dim_h * i1dim_w;
    int i1dim_hw  = i1dim_h * i1dim_w;

    float result = 0.0;

    for (int c = 0; c < i1dim_c; ++c) {
        int i1Index = b * i1dim_chw + c * i1dim_hw + y * i1dim_w + x;
        float val = input1[i1Index];
        result += val * val;
    }
    result = sqrt(result);
    output[index] = result;
}


__global__ void kernel_ChannelNorm_backward_input1(const int n, const float* input1, const long4 input1_size, const long4 input1_stride,
    const float* output, const long4 output_size, const long4 output_stride, const float* gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    float* gradInput, const long4 gradInput_size, const long4 gradInput_stride, int norm_deg) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    float val = 0.0;

    int dim_b = DIM0(gradInput_size);
    int dim_c = DIM1(gradInput_size);
    int dim_h = DIM2(gradInput_size);
    int dim_w = DIM3(gradInput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;


    int outIndex = b * dim_hw + y * dim_w + x;
    val = gradOutput[outIndex] * input1[index] / (output[outIndex]+1e-9);
    gradInput[index] = val;

}

void ChannelNorm_kernel_forward(THCState* state, THCudaTensor* input1, THCudaTensor* output, int norm_deg) {
    int n = 0;
    
    const long4 input1_size = make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]);
    const long4 input1_stride = make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]);

    const long4 output_size = make_long4(output->size[0], output->size[1], output->size[2], output->size[3]);
    const long4 output_stride = make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3]);

    n = THCudaTensor_nElement(state, output);
    kernel_ChannelNorm_updateOutput<<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        n, THCudaTensor_data(state, input1), input1_size, input1_stride, THCudaTensor_data(state, output), output_size, output_stride, 
        norm_deg);

    THCudaCheck(cudaGetLastError());
}

void ChannelNorm_kernel_backward(THCState* state, THCudaTensor* input1, THCudaTensor* output, THCudaTensor* gradOutput, THCudaTensor* gradInput1, int norm_deg) {
    int n = 0;

    const long4 input1_size = make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]);
    const long4 input1_stride = make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]);

    const long4 output_size = make_long4(output->size[0], output->size[1], output->size[2], output->size[3]);
    const long4 output_stride = make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3]);

    const long4 gradOutput_size = make_long4(gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
    const long4 gradOutput_stride = make_long4(gradOutput->stride[0], gradOutput->stride[1], gradOutput->stride[2], gradOutput->stride[3]);

    const long4 gradInput1_size = make_long4(gradInput1->size[0], gradInput1->size[1], gradInput1->size[2], gradInput1->size[3]);
    const long4 gradInput1_stride = make_long4(gradInput1->stride[0], gradInput1->stride[1], gradInput1->stride[2], gradInput1->stride[3]);

    n = THCudaTensor_nElement(state, gradInput1);
    kernel_ChannelNorm_backward_input1<<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        n, THCudaTensor_data(state, input1), input1_size, input1_stride, THCudaTensor_data(state, output), output_size, output_stride,
        THCudaTensor_data(state, gradOutput), gradOutput_size, gradOutput_stride, THCudaTensor_data(state, gradInput1), gradInput1_size, gradInput1_stride,
        norm_deg
    );

    THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
    }
#endif