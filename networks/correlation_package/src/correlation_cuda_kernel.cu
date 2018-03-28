#include <stdio.h>

#include "correlation_cuda_kernel.h"

#define real float

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

__global__ void channels_first(float* input, float* rinput, int channels, int height, int width, int pad_size)
{
    // n (batch size), c (num of channels), y (height), x (width)
    int n = blockIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.z;

    int ch_off = threadIdx.x;
    float value;

    int dimcyx = channels * height * width;
    int dimyx = height * width;

    int p_dimx = (width + 2 * pad_size);
    int p_dimy = (height + 2 * pad_size);
    int p_dimyxc = channels * p_dimy * p_dimx;
    int p_dimxc = p_dimx * channels;

    for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
      value = input[n * dimcyx + c * dimyx + y * width + x];
      rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
    }
}

__global__ void Correlation_forward( float *output, int nOutputChannels, int outputHeight, int outputWidth, 
                                     float *rInput1, int nInputChannels, int inputHeight, int inputWidth, 
                                     float *rInput2,
                                     int pad_size,
                                     int kernel_size,
                                     int max_displacement,
                                     int stride1,
                                     int stride2)
{
    // n (batch size), c (num of channels), y (height), x (width)
    
    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int n  = blockIdx.x;
    int y1 = blockIdx.y * stride1 + max_displacement + kernel_rad;
    int x1 = blockIdx.z * stride1 + max_displacement + kernel_rad;
    int c = threadIdx.x;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    float nelems = kernel_size * kernel_size * pdimc;

    __shared__ float prod_sum[THREADS_PER_BLOCK];

    // no significant speed-up in using chip memory for input1 sub-data, 
    // not enough chip memory size to accomodate memory per block for input2 sub-data
    // instead i've used device memory for both 

    // element-wise product along channel axis
    for (int tj = -displacement_rad; tj <= displacement_rad; ++tj ) {
      for (int ti = -displacement_rad; ti <= displacement_rad; ++ti ) {
        prod_sum[c] = 0;
        int x2 = x1 + ti*stride2;
        int y2 = y1 + tj*stride2;

        for (int j = -kernel_rad; j <= kernel_rad; ++j) {
          for (int i = -kernel_rad; i <= kernel_rad; ++i) {
            for (int ch = c; ch < pdimc; ch += THREADS_PER_BLOCK) {
              int indx1 = n * pdimyxc + (y1+j) * pdimxc + (x1 + i) * pdimc + ch;
              int indx2 = n * pdimyxc + (y2+j) * pdimxc + (x2 + i) * pdimc + ch;

              prod_sum[c] += rInput1[indx1] * rInput2[indx2];
            }
          }
        }

        // accumulate 
        __syncthreads();
        if (c == 0) {
          float reduce_sum = 0;
          for (int index = 0; index < THREADS_PER_BLOCK; ++index) {
            reduce_sum += prod_sum[index];
          }
          int tc = (tj + displacement_rad) * displacement_size + (ti + displacement_rad);
          const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx + blockIdx.z;
          output[tindx] = reduce_sum / nelems;
        }

      }
    }

}

__global__ void Correlation_backward_input1(int item, float *gradInput1, int nInputChannels, int inputHeight, int inputWidth, 
                                            float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth, 
                                            float *rInput2, 
                                            int pad_size,
                                            int kernel_size,
                                            int max_displacement,
                                            int stride1,
                                            int stride2)
  {
    // n (batch size), c (num of channels), y (height), x (width)

    int n = item; 
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;
    int tch_off = threadIdx.x;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int xmin = (x - kernel_rad - max_displacement) / stride1;
    int ymin = (y - kernel_rad - max_displacement) / stride1;

    int xmax = (x + kernel_rad - max_displacement) / stride1;
    int ymax = (y + kernel_rad - max_displacement) / stride1;

    if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
        // assumes gradInput1 is pre-allocated and zero filled
      return;
    }

    if (xmin > xmax || ymin > ymax) {
        // assumes gradInput1 is pre-allocated and zero filled
        return;
    }

    xmin = max(0,xmin);
    xmax = min(outputWidth-1,xmax);

    ymin = max(0,ymin);
    ymax = min(outputHeight-1,ymax);

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimcyx = nInputChannels * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    float nelems = kernel_size * kernel_size * nInputChannels;

    __shared__ float prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {

      int i2 = (tc % displacement_size - displacement_rad) * stride2;
      int j2 = (tc / displacement_size - displacement_rad) * stride2;

      int indx2 = n * pdimyxc + (y + j2)* pdimxc + (x + i2) * pdimc + c;
      
      float val2 = rInput2[indx2];

      for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
          int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
          prod_sum[tch_off] += gradOutput[tindx] * val2;
        }
      }
    }
    __syncthreads();

    if(tch_off == 0) {
      float reduce_sum = 0;
      for(int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
          reduce_sum += prod_sum[idx];
      }
      const int indx1 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
      gradInput1[indx1] = reduce_sum / nelems;
    }

}

__global__ void Correlation_backward_input2(int item, float *gradInput2, int nInputChannels, int inputHeight, int inputWidth,
                                            float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
                                            float *rInput1,
                                            int pad_size,
                                            int kernel_size,
                                            int max_displacement,
                                            int stride1,
                                            int stride2)
{
    // n (batch size), c (num of channels), y (height), x (width)

    int n = item;
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;

    int tch_off = threadIdx.x;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimcyx = nInputChannels * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    float nelems = kernel_size * kernel_size * nInputChannels;

    __shared__ float prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {
      int i2 = (tc % displacement_size - displacement_rad) * stride2;
      int j2 = (tc / displacement_size - displacement_rad) * stride2;

      int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
      int ymin = (y - kernel_rad - max_displacement - j2) / stride1;

      int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
      int ymax = (y + kernel_rad - max_displacement - j2) / stride1;

      if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
          // assumes gradInput2 is pre-allocated and zero filled
        continue;
      }

      if (xmin > xmax || ymin > ymax) {
          // assumes gradInput2 is pre-allocated and zero filled
          continue;
      }

      xmin = max(0,xmin);
      xmax = min(outputWidth-1,xmax);

      ymin = max(0,ymin);
      ymax = min(outputHeight-1,ymax);
      
      int indx1 = n * pdimyxc + (y - j2)* pdimxc + (x - i2) * pdimc + c;
      float val1 = rInput1[indx1];

      for (int j = ymin; j <= ymax; ++j) {
        for (int i = xmin; i <= xmax; ++i) {
          int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
          prod_sum[tch_off] += gradOutput[tindx] * val1;
        }
      }
    }

    __syncthreads();

    if(tch_off == 0) {
      float reduce_sum = 0;
      for(int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
          reduce_sum += prod_sum[idx];
      }
      const int indx2 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
      gradInput2[indx2] = reduce_sum / nelems;
    }

}

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
                                    /*THCState_getCurrentStream(state)*/ cudaStream_t stream)
{
   int batchSize = ob;

   int nInputChannels = ic;
   int inputWidth = iw;
   int inputHeight = ih;

   int nOutputChannels = oc;
   int outputWidth = ow;
   int outputHeight = oh;

   dim3 blocks_grid(batchSize, inputHeight, inputWidth);
   dim3 threads_block(THREADS_PER_BLOCK);

  channels_first<<<blocks_grid,threads_block, 0, stream>>> (input1,rInput1, nInputChannels, inputHeight, inputWidth,pad_size);
  channels_first<<<blocks_grid,threads_block, 0, stream>>> (input2,rInput2, nInputChannels, inputHeight, inputWidth, pad_size);

   dim3 threadsPerBlock(THREADS_PER_BLOCK);
   dim3 totalBlocksCorr(batchSize, outputHeight, outputWidth);

   Correlation_forward <<< totalBlocksCorr, threadsPerBlock, 0, stream >>> 
                        (output, nOutputChannels, outputHeight, outputWidth,
                         rInput1, nInputChannels, inputHeight, inputWidth,
                         rInput2,
                         pad_size,
                         kernel_size,
                         max_displacement,
                         stride1,
                         stride2);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Correlation_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}

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
                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{

    int batchSize = gob;
    int num = batchSize;

    int nInputChannels = ic;
    int inputWidth = iw;
    int inputHeight = ih;

    int nOutputChannels = goc;
    int outputWidth = gow;
    int outputHeight = goh;

    dim3 blocks_grid(batchSize, inputHeight, inputWidth);
    dim3 threads_block(THREADS_PER_BLOCK);

    channels_first<<<blocks_grid,threads_block, 0, stream>>> (input1, rInput1, nInputChannels,inputHeight, inputWidth, pad_size);
    channels_first<<<blocks_grid,threads_block, 0, stream>>> (input2, rInput2, nInputChannels, inputHeight, inputWidth, pad_size);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 totalBlocksCorr(inputHeight, inputWidth, nInputChannels);

    for (int n = 0; n < num; ++n) {
        Correlation_backward_input1 << <totalBlocksCorr, threadsPerBlock, 0, stream >> > (
            n, gradInput1, nInputChannels, inputHeight, inputWidth,
            gradOutput, nOutputChannels, outputHeight, outputWidth,
            rInput2,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);
    }

    for(int n = 0; n < batchSize; n++) {
        Correlation_backward_input2<<<totalBlocksCorr, threadsPerBlock, 0, stream>>>(
            n, gradInput2, nInputChannels, inputHeight, inputWidth,
            gradOutput, nOutputChannels, outputHeight, outputWidth,
            rInput1,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);
    }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Correlation_backward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}

#ifdef __cplusplus
}
#endif
