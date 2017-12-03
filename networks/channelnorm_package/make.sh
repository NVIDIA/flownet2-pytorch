#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling channelnorm kernels by nvcc..."
rm ChannelNorm_kernel.o
rm -r ../_ext

nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

cd ../
python build.py