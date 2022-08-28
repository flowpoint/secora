#!/bin/bash
if command -v docker &> /dev/null
then
	echo using docker;
	container_command=docker
elif command -v podman &> /dev/null
then
    echo using podman;
	container_command=podman
fi

$container_command run -it \
    --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /home/slow4/huggingface:/root/.cache/huggingface:z \
    -v /home/slow4/secora_output:/root/secora_output:z \
    -v /home/flowpoint/secora:/root/secora:z \
    flowpoint/secora_dev:0.0.1 \
    /bin/bash
