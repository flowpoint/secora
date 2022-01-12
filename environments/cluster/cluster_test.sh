#!/bin/bash

#SBATCH --partition=wr14          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=20G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=40:00:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=logs/slurm_tests.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=logs/slurm_test.%j.err     # filename for STDERR

cd ~/secora
# setup
module load cmake
module load cuda/11.2
module load python3

export HF_HOME=/scratch/fhoels2s/huggingface

#debugging flags
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

#unset NCCL_SOCKET_IFNAME

#workaround for thread-unsafe tokenizers:
export TOKENIZERS_PARALLELISM=false
pipenv run python secora/train.py
