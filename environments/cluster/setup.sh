cd ~/secora
# setup
#module load gcc/9.1.0
module load gcc/8.2.0
module load cmake
module load cuda/11.2
module load python3

#building torch 1.10.1 for cuda 11.2

#optional
#pipenv shell
pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

#export CUDA_HOME=/usr/local/cuda
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
#export PATH=$PATH:$CUDA_HOME/bin

export CUDA_HOME=/usr/local/cuda-11.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive 

USE_MKLDNN=0
export USE_GLOO=0
export BUILD_CAFFE2=0
export BUILD_CAFFE2_OPS=0
export BUILD_TEST=0
export TORCH_CUDA_ARCH_LIST='3.7;7.0'
#python setup.py build --cmake-only > build_log.out 2> build_log.err
python setup.py install > build_log.out 2> build_log.err

