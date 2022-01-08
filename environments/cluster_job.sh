#!/bin/bash

#SBATCH --partition=any

# use miniconda?
#wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
#chmod u+x Miniconda3...
#./Miniconda3../bin/conda
#~/miniconda/bin/conda install -c pytorch faiss-cpu

#SBATCH --partition=gpu          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=64                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=5:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR


# setup
module load cmake
module load cuda/11.2
module load python3

pip3 install pipenv
pipenv --python 3.9.2 shell
pipenv install

pip3 install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

#pip3 install faiss-cpu


