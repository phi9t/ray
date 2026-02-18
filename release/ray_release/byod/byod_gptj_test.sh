#!/bin/bash

set -exo pipefail

uv pip install --system -c "$HOME/requirements_compiled.txt" myst-parser myst-nb
