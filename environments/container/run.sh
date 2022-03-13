#!/bin/bash

if (("$(id -u)" != 0))
then
	echo this script is expected to be run as sudo, because rocm needs acess to /dev/.
	exit -1
fi

if command -v docker &> /dev/null
then
	echo using docker;
	container_command=docker
elif command -v podman &> /dev/null
then
    echo using podman;
	container_command=podman
fi

$container_command run -it --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /home/fast4/huggingface:/root/.cache/huggingface:z \
    -v ../secora:/root/secora:z \
    flowpoint/secora_dev /bin/bash
