import os
import torch
import torch.utils.ffi

this_folder = os.path.dirname(os.path.abspath(__file__)) + '/'

Headers = []
Sources = []
Defines = []
Objects = []

if torch.cuda.is_available() == True:
    Headers += ['src/Resample2d_cuda.h']
    Sources += ['src/Resample2d_cuda.c']
    Defines += [('WITH_CUDA', None)]
    Objects += ['src/Resample2d_kernel.o']

ffi = torch.utils.ffi.create_extension(
    name='_ext.resample2d',
    headers=Headers,
    sources=Sources,
    verbose=False,
    with_cuda=True,
    package=False,
    relative_to=this_folder,
    define_macros=Defines,
    extra_objects=[os.path.join(this_folder, Object) for Object in Objects]
)

if __name__ == '__main__':
    ffi.build()