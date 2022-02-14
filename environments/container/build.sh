#!/bin/bash
dir=$(dirname ${BASH_SOURCE[0]})

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

$container_command build --label secora_dev --tag flowpoint/secora_dev:0.0.1 -f "$dir/Dockerfile" "$dir"
