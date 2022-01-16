#!/bin/bash
echo "syncing cluster_output to ../cluster_output"
rsync -a --progress cluster_wr0:~/cluster_output/ ../cluster_output/
echo "syncing cluster slurm_logs to ../slurm_logs"
rsync -a --progress cluster_wr0:~/slurm_logs ../slurm_logs

echo "syncing secora to cluster"
rsync -a \
    --delete-after \
    --exclude output \
    --exclude output2 \
    --exclude third_party \
    --exclude .git \
    --exclude Pipfile.lock \
    --progress \
    ../secora \
    cluster_wr0:~


