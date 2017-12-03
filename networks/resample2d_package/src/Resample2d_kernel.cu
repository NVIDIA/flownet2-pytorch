#include <THC.h>
#include <THCGeneral.h>
#include <stdbool.h>
#include <stdio.h>

#define CUDA_NUM_THREADS 512 
#define THREADS_PER_BLOCK 64 

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
    extern "C" {
#endif

__global__ void kernel_Resample2d_updateOutput(
    const int n,
    const float* input1, const long4 input1_size, const long4 input1_stride,
    const float* input2, const long4 input2_size, const long4 input2_stride,
    float* output, const long4 output_size, const long4 output_stride, 
    int kernel_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    float val = 0.0;

    int dim_b = VEC_0(output_size);
    int dim_c = VEC_1(output_size);
    int dim_h = VEC_2(output_size);
    int dim_w = VEC_3(output_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    float dx = IDX_4(input2, b, 0, y, x);
    float dy = IDX_4(input2, b, 1, y, x);

    float xf = float(x) + dx;
    float yf = float(y) + dy;
    float alpha = xf - floor(xf); // alpha
    float beta = yf - floor(yf); // beta

    int xL = max(min( int (floor(xf)),    dim_w-1), 0);
    int xR = max(min( int (floor(xf)+1), dim_w -1), 0);
    int yT = max(min( int (floor(yf)),    dim_h-1), 0);
    int yB = max(min( int (floor(yf)+1),  dim_h-1), 0);

    for (int fy = 0; fy < kernel_size; fy += 1) {
        for (int fx = 0; fx < kernel_size; fx += 1) {
            val += (1. - alpha)*(1. - beta) * IDX_4(input1, b, c, yT + fy, xL + fx);
            val +=    (alpha)*(1. - beta) * IDX_4(input1, b, c, yT + fy, xR + fx);
            val +=    (1. - alpha)*(beta) * IDX_4(input1, b, c, yB + fy, xL + fx);
            val +=       (alpha)*(beta) * IDX_4(input1, b, c, yB + fy, xR + fx);
        }
    }

    output[index] = val;

}


__global__ void kernel_Resample2d_backward_input1(
    const int n,
    const float* input1, const long4 input1_size, const long4 input1_stride,
    const float* input2, const long4 input2_size, const long4 input2_stride,
    const float* gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    float* gradInput, const long4 gradInput_size, const long4 gradInput_stride, 
    int kernel_size
) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int dim_b = VEC_0(gradOutput_size);
    int dim_c = VEC_1(gradOutput_size);
    int dim_h = VEC_2(gradOutput_size);
    int dim_w = VEC_3(gradOutput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    float dx = IDX_4(input2, b, 0, y, x);
    float dy = IDX_4(input2, b, 1, y, x);

    float xf = float(x) + dx;
    float yf = float(y) + dy;
    float alpha = xf - int(xf); // alpha
    float beta = yf - int(yf); // beta

    int idim_h = VEC_2(input1_size);
    int idim_w = VEC_3(input1_size);

    int xL = max(min( int (floor(xf)),    idim_w-1), 0);
    int xR = max(min( int (floor(xf)+1), idim_w -1), 0);
    int yT = max(min( int (floor(yf)),    idim_h-1), 0);
    int yB = max(min( int (floor(yf)+1),  idim_h-1), 0);

    for (int fy = 0; fy < kernel_size; fy += 1) {
        for (int fx = 0; fx < kernel_size; fx += 1) {
            atomicAdd(&IDX_4(gradInput, b, c, (yT + fy), (xL + fx)), (1-alpha)*(1-beta) * IDX_4(gradOutput, b, c, y, x));
            atomicAdd(&IDX_4(gradInput, b, c, (yT + fy), (xR + fx)),   (alpha)*(1-beta) * IDX_4(gradOutput, b, c, y, x));
            atomicAdd(&IDX_4(gradInput, b, c, (yB + fy), (xL + fx)),   (1-alpha)*(beta) * IDX_4(gradOutput, b, c, y, x));
            atomicAdd(&IDX_4(gradInput, b, c, (yB + fy), (xR + fx)),     (alpha)*(beta) * IDX_4(gradOutput, b, c, y, x));
        }
    }

}

__global__ void kernel_Resample2d_backward_input2(
    const int n,
    const float* input1, const long4 input1_size, const long4 input1_stride,
    const float* input2, const long4 input2_size, const long4 input2_stride,
    const float* gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    float* gradInput, const long4 gradInput_size, const long4 gradInput_stride, 
    int kernel_size

) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    float output = 0.0;
    int kernel_rad = (kernel_size - 1)/2;

    int dim_b = VEC_0(gradInput_size);
    int dim_c = VEC_1(gradInput_size);
    int dim_h = VEC_2(gradInput_size);
    int dim_w = VEC_3(gradInput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    int odim_c = VEC_1(gradOutput_size);

    float dx = IDX_4(input2, b, 0, y, x);
    float dy = IDX_4(input2, b, 1, y, x);

    float xf = float(x) + dx;
    float yf = float(y) + dy;

    int xL = max(min( int (floor(xf)),    dim_w-1), 0);
    int xR = max(min( int (floor(xf)+1), dim_w -1), 0);
    int yT = max(min( int (floor(yf)),    dim_h-1), 0);
    int yB = max(min( int (floor(yf)+1),  dim_h-1), 0);
    
    if (c % 2) {
        float gamma = 1 - (xf - floor(xf)); // alpha
        for (int i = 0; i <= 2*kernel_rad; ++i) {
            for (int j = 0; j <= 2*kernel_rad; ++j) {
                for (int ch = 0; ch < odim_c; ++ch) {
                    output += (gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yB + j), (xL + i));
                    output -= (gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yT + j), (xL + i));
                    output += (1-gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yB + j), (xR + i));
                    output -= (1-gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yT + j), (xR + i));
                }
            }
        }
    }
    else {
        float gamma = 1 - (yf - floor(yf)); // alpha
        for (int i = 0; i <= 2*kernel_rad; ++i) {
            for (int j = 0; j <= 2*kernel_rad; ++j) {
                for (int ch = 0; ch < odim_c; ++ch) {
                    output += (gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yT + j), (xR + i));
                    output -= (gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yT + j), (xL + i));
                    output += (1-gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yB + j), (xR + i));
                    output -= (1-gamma) * IDX_4(gradOutput, b, ch, y, x) * IDX_4(input1, b, ch, (yB + j), (xL + i));
                }
            }
        }

    }

    gradInput[index] = output;

}

void Resample2d_kernel_forward(
    THCState* state,
    THCudaTensor* input1,
    THCudaTensor* input2,
    THCudaTensor* output,
    int kernel_size
) {
    int n = 0;

    n = THCudaTensor_nElement(state, output);
    kernel_Resample2d_updateOutput<<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        n,
        THCudaTensor_data(state, input1), make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]), make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]),
        THCudaTensor_data(state, input2), make_long4(input2->size[0], input2->size[1], input2->size[2], input2->size[3]), make_long4(input2->stride[0], input2->stride[1], input2->stride[2], input2->stride[3]),
        THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3]), 
        kernel_size
    );

    THCudaCheck(cudaGetLastError());
}

void Resample2d_kernel_backward(
    THCState* state,
    THCudaTensor* input1,
    THCudaTensor* input2,
    THCudaTensor* gradOutput, 
    THCudaTensor* gradInput1,
    THCudaTensor* gradInput2, 
    int kernel_size

) {
    int n = 0;

    n = THCudaTensor_nElement(state, gradOutput);
    kernel_Resample2d_backward_input1<<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        n,
        THCudaTensor_data(state, input1), make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]), make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]),
        THCudaTensor_data(state, input2), make_long4(input2->size[0], input2->size[1], input2->size[2], input2->size[3]), make_long4(input2->stride[0], input2->stride[1], input2->stride[2], input2->stride[3]),
        THCudaTensor_data(state, gradOutput), make_long4(gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]), make_long4(gradOutput->stride[0], gradOutput->stride[1], gradOutput->stride[2], gradOutput->stride[3]),
        THCudaTensor_data(state, gradInput1), make_long4(gradInput1->size[0], gradInput1->size[1], gradInput1->size[2], gradInput1->size[3]), make_long4(gradInput1->stride[0], gradInput1->stride[1], gradInput1->stride[2], gradInput1->stride[3]),
        kernel_size
    );

    n = THCudaTensor_nElement(state, gradInput2);
    kernel_Resample2d_backward_input2<<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        n,
        THCudaTensor_data(state, input1), make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]), make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]),
        THCudaTensor_data(state, input2), make_long4(input2->size[0], input2->size[1], input2->size[2], input2->size[3]), make_long4(input2->stride[0], input2->stride[1], input2->stride[2], input2->stride[3]),
        THCudaTensor_data(state, gradOutput), make_long4(gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]), make_long4(gradOutput->stride[0], gradOutput->stride[1], gradOutput->stride[2], gradOutput->stride[3]),
        THCudaTensor_data(state, gradInput2), make_long4(gradInput2->size[0], gradInput2->size[1], gradInput2->size[2], gradInput2->size[3]), make_long4(gradInput2->stride[0], gradInput2->stride[1], gradInput2->stride[2], gradInput2->stride[3]), 
        kernel_size
    );
    THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
    }
#endif