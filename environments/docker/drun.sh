#!/bin/bash
sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v /home/fast4/huggingface:/root/.cache/huggingface:z flowpoint/rocm_autotrain /bin/bash
