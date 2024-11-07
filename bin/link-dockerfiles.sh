#!/bin/bash

if docker info -f "{{println .SecurityOptions}}" | grep rootless > /dev/null ; then
    ln -snfv docker/rootless/docker-compose.yml docker-compose.yml
    ln -snfv docker/rootless/Dockerfile Dockerfile
    ln -snfv docker/rootless/requirements.txt requirements.txt
elif [[ "$(uname)" == "Darwin" ]] ; then
    ln -snfv docker/no-nvidia-devices/docker-compose.yml docker-compose.yml
    ln -snfv docker/no-nvidia-devices/Dockerfile Dockerfile
    ln -snfv docker/no-nvidia-devices/requirements.txt requirements.txt
else
    ln -snfv docker/nvidia-uid-1000/docker-compose.yml docker-compose.yml
    ln -snfv docker/nvidia-uid-1000/Dockerfile Dockerfile
    ln -snfv docker/nvidia-uid-1000/requirements.txt requirements.txt
fi