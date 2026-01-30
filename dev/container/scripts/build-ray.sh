#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/venv/bin:/root/.local/bin:$PATH"
export VIRTUAL_ENV=/opt/venv

cd /workspace/ray/python

# Pre-install thirdparty deps that setup.py tries to install via --target
pip install -q --upgrade --target=/workspace/ray/python/ray/thirdparty_files \
    psutil "setproctitle==1.2.2" colorama

# Skip Bazel rebuild if native binaries already exist
if [[ -f /workspace/ray/python/ray/_raylet.so ]]; then
    echo "Native binaries found, skipping Bazel build (set SKIP_BAZEL_BUILD=0 to force)"
    export SKIP_BAZEL_BUILD=${SKIP_BAZEL_BUILD:-1}
fi

pip install -e ".[default]" --no-build-isolation --verbose
echo "Ray build complete: $(python -c 'import ray; print(ray.__version__)')"
