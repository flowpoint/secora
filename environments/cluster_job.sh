#!/bin/bash

#SBATCH --partition=any

# use miniconda?
#wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
#chmod u+x Miniconda3...
#./Miniconda3../bin/conda
#~/miniconda/bin/conda install -c pytorch faiss-cpu

module load cmake
module load cuda/11.2
module load python3

pip3 install faiss-cpu

