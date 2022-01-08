#!/bin/bash

tar -cvf \
    ../secora.tar \
    --exclude secora/output \
    --exclude secora/third_party \
    --exclude-vcs-ignores \
    --exclude-vcs \
    ../secora

scp ../secora.tar cluster:~
