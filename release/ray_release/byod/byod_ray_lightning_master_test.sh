#!/bin/bash
# This script is used to build an extra layer on top of the base anyscale/ray image
# to run the agent stress test.

set -exo pipefail

uv pip install --system -U --force-reinstall pytorch-lightning lightning-bolts
pip uninstall ray_lightning -y # Uninstall first so pip does a reinstall.
uv pip install --system -U --no-cache-dir git+https://github.com/ray-project/ray_lightning#ray_lightning
uv pip install --system --force-reinstall torch==1.11.0
uv pip install --system --force-reinstall torchvision==0.12.0
