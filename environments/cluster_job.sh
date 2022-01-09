#!/bin/bash

#SBATCH --partition=gpu          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=100G                 # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=5:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

cd ~/secora
# setup
module load cmake
module load cuda/11.2
module load python3

pipenv run python secora/train.py
