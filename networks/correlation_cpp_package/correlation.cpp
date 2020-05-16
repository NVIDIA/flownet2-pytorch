/*
Copyright 2020 Samim Taray

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
*/
#include <torch/extension.h>
using namespace torch;

#include <iostream>

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)
#define WITHIN_BOUNDS3(val1, val2, bound1, bound2) (val1 >= 0 && val1 < bound1 && val2 >= 0 && val2 < bound2)
#define WITHIN_BOUNDS2(x, bound) (x >=0 && x < bound)


template <typename scalar_t>
static void correlate_patch(
    TensorAccessor<scalar_t, 3> f1_acc,
    TensorAccessor<scalar_t, 3> f2_acc,
    TensorAccessor<scalar_t, 3> out_acc,
    int n, /* batch number */
    int h, /* height cordinate number */
    int w, /* width cordinate number */
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2
){
    /*
    Algorithm: we are in the h,w position of both feature maps. Where do we go from here?
    Let's see. 
    */
    int f1c = f1_acc.size(0);
    int f1h = f1_acc.size(1);
    int f1w = f1_acc.size(2);

    /* Indicies that define the extents of the window. */
    int win_starth, win_endh, win_startw, win_endw; 
    win_starth = h - max_displacement;
    win_endh = h + max_displacement + 1;
    win_startw = w - max_displacement;
    win_endw = w + max_displacement + 1;

    int c, ph, pw, outpc = 0;
    
    for ( ph = win_starth; ph < win_endh; ph+=stride2){
        for ( pw = win_startw; pw < win_endw; pw+=stride2){
            if ( WITHIN_BOUNDS3(ph, pw, f1h, f1w /* better to have f2 here maybe */)){
                // We are in the window now.
                scalar_t outval = 0.0;
                for (c = 0; c < f1c; c++){
                    outval += f1_acc[c][h][w] * f2_acc[c][ph][pw];
                }
                // TODO: Optimization: We can get this from ph and pw. This should be ph * (win_endh - win_starth)
                // outpc = (( ph - win_starth)/stride2) * max_displacement + (pw - win_startw)/stride2; // Output channel index.
                // std::cout<<"outpc = "<<outpc<<std::endl;
                scalar_t *access_ptr = &out_acc[outpc][h][w];
                *access_ptr = outval / scalar_t(c);
            }
            outpc++;
        }
    }
}

torch::Tensor correlation_cpp_forward(
    torch::Tensor f1,
    torch::Tensor f2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2){

        const auto f1b = f1.size(0);
        const auto f1h = f1.size(2);
        const auto f1w = f1.size(3);        

        // Works as long as stride2 and max_displacement agree.
        const auto outc = ( 2*(max_displacement / stride2) + 1) * ( 2*(max_displacement / stride2) + 1);
        const auto outb = f1b;
        const auto outh = f1h;
        const auto outw = f1w;

        auto output = at::zeros({outb, outc, outh, outw}, f1.options());

        int n, h, w;
        #pragma omp parallel for private(n, h, w) collapse(2)
        for (n = 0; n < outb; ++n) {
            for(h = 0; h < outh; ++h){
                for(w = 0; w < outw; ++w){
                    AT_DISPATCH_FLOATING_TYPES(f1.scalar_type(), "correlation_forward_cpp", ([&]{
                        auto f1_acc = f1.accessor<scalar_t, 4>();
                        auto f2_acc = f2.accessor<scalar_t, 4>();
                        auto out_acc = output.accessor<scalar_t, 4>();

                        correlate_patch(
                                f1_acc[n],
                                f2_acc[n],
                                out_acc[n],
                                n,
                                h,
                                w,
                                pad_size,
                                kernel_size,
                                max_displacement,
                                stride1,
                                stride2);
                    }));
                }
            }
        }
        return output;
    }



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_cpp_forward, "Spatial Correlation Sampler Forward");
//   m.def("backward", &correlation_cpp_backward, "Spatial Correlation Sampler backward");
}