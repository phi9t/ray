#!/bin/bash

set -exo pipefail

uv pip install --system -U "gymnasium[mujoco]"==1.1.1 ale_py==0.10.1 imageio==2.34.2 opencv-python-headless==4.9.0.80
uv pip install --system -U torch==2.7 torchvision==0.22 --index-url https://download.pytorch.org/whl/cu128
uv pip install --system -U pettingzoo==1.24.3
uv pip install --system -U pygame wandb
