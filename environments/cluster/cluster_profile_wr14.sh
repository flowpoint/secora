#!/bin/bash

#SBATCH --partition=wr14          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=12     # number of tasks per node
#SBATCH --mem=48G                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=20:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm_logs/slurm_profile.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm_logs/slurm_profile.%j.err     # filename for STDERR

cd ~/secora
# setup
module load gcc/8.2.0
module load cmake
module load cuda/11.2
module load python3

export HF_HOME=/scratch/fhoels2s/huggingface
export MASTER_ADDR="localhost"
export MASTER_PORT=$((15000 + $RANDOM % 5000))

#debugging flags
unset TORCH_DISTRIBUTED_DEBUG=DETAIL
unset NCCL_DEBUG_SUBSYS
unset NCCL_DEBUG
unset NCCL_IB_DISABLE=1

#unset NCCL_SOCKET_IFNAME

#workaround for thread-unsafe tokenizers:
export TOKENIZERS_PARALLELISM=false
pipenv run python secora/profiling.py configs/cluster.yml --modes train --name profile_wr14_train --batch_size 8

pipenv run python secora/profiling.py configs/cluster.yml --modes embedding --name profile_wr14_embedding --batch_size 8

pipenv run python secora/profiling.py configs/cluster.yml --modes validation --name profile_wr14_validation --batch_size 8
