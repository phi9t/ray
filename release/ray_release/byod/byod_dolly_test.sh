#!/bin/bash
# This script is used to build an extra layer on top of the base anyscale/ray image
# to run the agent stress test.

set -exo pipefail

uv pip install --system -c "$HOME/requirements_compiled.txt" myst-parser myst-nb

pip3 uninstall -y pytorch-lightning
uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

uv pip install --system lightning==2.0.3
