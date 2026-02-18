#!/bin/bash

set -exo pipefail

sudo apt-get update -y
sudo apt-get install --no-install-recommends -y libgl1-mesa-glx libmagic1 poppler-utils tesseract-ocr libreoffice
sudo rm -f /etc/apt/sources.list.d/*

# Install runtime deps
uv pip install --system "unstructured[all-docs]==0.18.21"
uv pip install --system --force-reinstall --no-cache-dir pandas==2.3.3
