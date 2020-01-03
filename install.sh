#!/bin/bash
cd ./networks/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install
cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install
cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install
cd ..
