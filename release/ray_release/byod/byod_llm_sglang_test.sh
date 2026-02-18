#!/bin/bash
# This script is used to build an extra layer on top of the base llm image
# to run the llm sglang release tests

set -exo pipefail

uv pip install --system "sglang[all]==0.5.6.post1"
