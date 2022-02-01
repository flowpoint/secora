#!/bin/bash

#SBATCH --partition=gpu4          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=96G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=00:50:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm_logs/slurm_debug.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm_logs/slurm_debug.%j.err     # filename for STDERR

cd ~/secora
# setup
module load gcc/8.2.0
module load cmake
module load cuda/11.2
module load python3

export HF_HOME=/scratch/fhoels2s/huggingface

#debugging flags
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

export CUDA_LAUNCH_BLOCKING=1

#unset NCCL_SOCKET_IFNAME

#workaround for thread-unsafe tokenizers:
export TOKENIZERS_PARALLELISM=false
#pipenv run python secora/train.py --debug configs/cluster.yml --batch_size 8 --run_name debug_cluster_gpu4_8

pipenv run python secora/train.py --debug configs/cluster.yml --run_name debug_cluster_gpu4_18

#pipenv run python secora/train.py --debug configs/cluster.yml --batch_size 24 --run_name debug_cluster_gpu4_10

#pipenv run python secora/train.py --debug configs/cluster.yml --batch_size 32 --run_name debug_cluster_gpu4_11
