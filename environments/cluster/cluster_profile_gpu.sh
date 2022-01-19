#!/bin/bash

#SBATCH --partition=gpu          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=8      # number of tasks per node
#SBATCH --mem=24G                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=20:00             # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm_logs/slurm_profile.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm_logs/slurm_profile.%j.err     # filename for STDERR

cd ~/secora
# setup
module load gcc/8.2.0
module load cmake
module load cuda/11.2
module load python3

export HF_HOME=/scratch/fhoels2s/huggingface

#debugging flags
unset TORCH_DISTRIBUTED_DEBUG=DETAIL
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG
unset NCCL_IB_DISABLE=1

#unset NCCL_SOCKET_IFNAME

#workaround for thread-unsafe tokenizers:
export TOKENIZERS_PARALLELISM=false
pipenv run python secora/profiling.py configs/cluster.yml --modes train --run_name profile_cluster_train20 --batch_size 64

pipenv run python secora/profiling.py configs/cluster.yml --modes embedding --run_name profile_cluster_embedding20 --batch_size 64

pipenv run python secora/profiling.py configs/cluster.yml --modes validation --run_name profile_cluster_validation20 --batch_size 64
