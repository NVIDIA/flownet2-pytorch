# flownet2-pytorch 

Pytorch implementation of [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925).<br /> FlowNet2 Caffe implementation : [flownet2](https://github.com/lmb-freiburg/flownet2)

Multiple GPU training is supported, and the code provides examples for training or inference on [MPI-Sintel](http://sintel.is.tue.mpg.de/) clean and final datasets. The same commands can be used for training or inference with other datasets. See below for more detail.

Inference using fp16 (half-precision) is also supported.

For more help, type <br />
    
    python main.py --help

## Network architectures
Below are the different flownet neural network architectures that are provided. <br />
A batchnorm version for each network is available.

 - **FlowNet2S**
 - **FlowNet2C**
 - **FlowNet2CS**
 - **FlowNet2CSS**
 - **FlowNet2SD**
 - **FlowNet2**

## Custom layers

`FlowNet2` or `FlowNet2C*` achitectures rely on custom layers `Resample2d` or `Correlation`. <br />
A pytorch implementation of these layers with cuda kernels are available at [./networks](./networks). <br />
Note : Currently, half precision kernels are not available for these layers.

## Data Loaders

Dataloaders for FlyingChairs, FlyingThings, ChairsSDHom and ImagesFromFolder are available in [datasets.py](./datasets.py). <br />

## Loss Functions

L1 and L2 losses with multi-scale support are available in [losses.py](./losses.py). <br />

## Installation 

    # get flownet2-pytorch source
    git clone https://github.com/NVIDIA/flownet2-pytorch.git
    cd flownet2-pytorch

    # install custom layers
    bash install.sh

## Docker image
Libraries and other dependencies for this project include: Ubuntu 16.04, Python 2.7, Pytorch 0.2, CUDNN 6.0, CUDA 8.0

A Dockerfile with the above dependencies is available 
    
    # Build and launch docker image
    bash launch_docker.sh

## Convert Official [Caffe Pre-trained Models](https://lmb.informatik.uni-freiburg.de/resources/software.php) to PyTorch
1. Download caffe-models and build flownet2-caffe. This step creates a docker image named `flownet2`. The pre-trained caffe-models will be located in `/flownet2/flownet2/models` inside this container. In total, this step will require ~8.5 GB of memory. More information on how the caffe models are pre-trained can be found here : [https://lmb.informatik.uni-freiburg.de/resources/software.php](https://lmb.informatik.uni-freiburg.de/resources/software.php)
    
    #Run this command for step 1 <br />
    bash ./download_caffe_models.sh
    
 2. Convert caffe-models to PyTorch. This step will launch flownet2 image, and convert and save PyTorch checkpoints. The input argument to this step is the path of your cloned flownet2-pytorch.

    #Run this command for step 2 <br />
    bash ./run-caffe2pytorch.sh /path/to/your/flownet2-pytorch/clone

The above steps will create checkpoints for the different architectures named as below: <br />
    
    FlowNet2_checkpoint.pth.tar 
    FlowNet2-C_checkpoint.pth.tar 
    FlowNet2-CS_checkpoint.pth.tar 
    FlowNet2-CSS_checkpoint.pth.tar 
    FlowNet2-CSS-ft-sd_checkpoint.pth.tar 
    FlowNet2-S_checkpoint.pth.tar 
    FlowNet2-SD_checkpoint.pth.tar 
    
## Inference
    # Example on MPISintel Clean   
    python main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean \
    --inference_dataset_root /path/to/mpi-sintel/clean/dataset \
    --resume /path/to/checkpoints 
    
## Training and validation

    # Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
    python main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
    --training_dataset MpiSintelFinal --training_dataset_root /path/to/mpi-sintel/final/dataset  \
    --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset

    # Example on MPISintel Final and Clean, with MultiScale loss on FlowNet2C model 
    python main.py --batch_size 8 --model FlowNet2C --optimizer=Adam --optimizer_lr=1e-4 --loss=MultiScale --loss_norm=L1 \
    --loss_numScales=5 --loss_startScale=4 --optimizer_lr=1e-4 --crop_size 384 512 \
    --training_dataset FlyingChairs --training_dataset_root /path/to/flying-chairs/dataset  \
    --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset
    
## Results on MPI-Sintel
[![Predicted flows on MPI-Sintel](./image.png)](https://www.youtube.com/watch?v=HtBmabY8aeU "Predicted flows on MPI-Sintel")

## Acknowledgments
Parts of this code were derived, as noted in the code, from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch).
