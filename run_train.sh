#!/bin/bash

#PBS -N Spirou_GPU_Training
#PBS -q comm_gpu
#PBS -l gpus=1



module load singularity/2.4.5
cd ..
cd ..
cd scratch/
cd ljm0015/
echo $PWD
cd Cluster/
cd images/
singularity shell --nv keras-full.img
cd ..
cd u-net-keras/
python2 train.py 1