#!/bin/bash
# shellcheck disable=SC2102

set -exo pipefail

uv pip install --system --no-cache-dir pybase64==1.4.2
