#include <TH/TH.h>

int Correlation_forward_cpu(THFloatTensor *input1,
                      THFloatTensor *input2,
                      THFloatTensor *rInput1,
                      THFloatTensor *rInput2,
                      THFloatTensor *output,
                      int pad_size,
                      int kernel_size,
                      int max_displacement,
                      int stride1,
                      int stride2,
                      int corr_type_multiply)
{
    return 1;
}

int Correlation_backward_cpu(THFloatTensor *input1,
                       THFloatTensor *input2,
                       THFloatTensor *rInput1,
                       THFloatTensor *rInput2,
                       THFloatTensor *gradOutput,
                       THFloatTensor *gradInput1,
                       THFloatTensor *gradInput2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{
    return 1;
}
