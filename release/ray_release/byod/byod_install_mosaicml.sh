#!/bin/bash
# shellcheck disable=SC2102

set -exo pipefail

uv pip install --system mosaicml-streaming==0.5.1
