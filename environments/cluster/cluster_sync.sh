#!/bin/bash

rsync -a --progress cluster_wr0:~/cluster_output ../cluster_output 
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


