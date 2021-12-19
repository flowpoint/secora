#!/bin/bash
sudo docker build --label rocm_autotrain --rm  --tag flowpoint/rocm_autotrain:0.0.1 -f environments/docker/Dockerfile . 
#docker/
