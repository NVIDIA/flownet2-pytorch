#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling resample2d kernels by nvcc..."
rm Resample2d_kernel.o
rm -r ../_ext

nvcc -c -o Resample2d_kernel.o Resample2d_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

cd ../
python build.py