#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src

echo "Compiling correlation kernels by nvcc..."

rm correlation_cuda_kernel.o
rm -r ../_ext

# nvcc -c -o correlation_cuda_kernel.o correlation_cuda_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC
nvcc -c -o correlation_cuda_kernel.o correlation_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
