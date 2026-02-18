#!/bin/bash

cd "${0%/*}" || exit 1

sudo apt update
sudo apt -y install build-essential
uv pip install --system cmake

uv pip install --system -U -r ./driver_requirements.txt


HOROVOD_WITH_GLOO=1 HOROVOD_WITHOUT_MPI=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_PYTORCH=1 uv pip install --system horovod
