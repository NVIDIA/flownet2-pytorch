#!/bin/bash
cd ./networks/correlation_package
python setup.py install
cd ../resample2d_package 
python setup.py install
cd ../channelnorm_package 
python setup.py install
cd ..
