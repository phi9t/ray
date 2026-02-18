#!/bin/bash
# This script is used to build an extra layer on top of the base anyscale/ray image
# to run the agent stress test.

set -exo pipefail

uv pip install --system -U --force-reinstall ray-lightning pytorch-lightning lightning-bolts
uv pip install --system --force-reinstall torch==1.11.0
uv pip install --system --force-reinstall torchvision==0.12.0
