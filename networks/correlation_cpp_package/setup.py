from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='correlation',
    ext_modules=[
        cpp_extension.CppExtension(
            'correlation', # Name of the module used in pybind
            ['correlation.cpp'], #  source files
            extra_compile_args={'cxx': ['-fopenmp']},
            extra_link_args=['-lgomp'])
    ],
    author='Samim Zahoor Taray',
    author_email='zsameem@gmail.com',
    install_requires=['torch>=1.1', 'numpy'],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)