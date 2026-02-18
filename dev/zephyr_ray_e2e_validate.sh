#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ZEPHYR_IMAGE:-ghcr.io/phi9t/sygaldry/zephyr:sglang-miles-dev}"
BAZEL_CACHE_HOST="${BAZEL_CACHE_HOST:-/mnt/data_infra/zephyr_container_infra/shared/bazel_cache}"
UV_CACHE_HOST="${UV_CACHE_HOST:-/mnt/data_infra/zephyr_container_infra/shared/uv_cache}"
BAZELISK_CACHE_HOST="${BAZELISK_CACHE_HOST:-/mnt/data_infra/zephyr_container_infra/shared/bazelisk_cache}"
BAZEL_OUTPUT_ROOT="${BAZEL_OUTPUT_ROOT:-/mnt/shared/bazel_cache/ray-sglang-codex-build}"
UV_CACHE_INNER="${UV_CACHE_INNER:-/mnt/shared/uv_cache/ray-install}"
BAZELISK_HOME_INNER="${BAZELISK_HOME_INNER:-/mnt/shared/bazelisk}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
RAY_EXCLUDES_FILE="${RAY_EXCLUDES_FILE:-/tmp/ray_uv_excludes.txt}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required" >&2
  exit 1
fi

if [[ ! -d "${BAZEL_CACHE_HOST}" ]]; then
  echo "ERROR: missing bazel cache dir: ${BAZEL_CACHE_HOST}" >&2
  exit 1
fi

if [[ ! -d "${UV_CACHE_HOST}" ]]; then
  echo "ERROR: missing uv cache dir: ${UV_CACHE_HOST}" >&2
  exit 1
fi

if [[ ! -d "${BAZELISK_CACHE_HOST}" ]]; then
  echo "ERROR: missing bazelisk cache dir: ${BAZELISK_CACHE_HOST}" >&2
  exit 1
fi

echo "[0/4] Normalize workspace ownership for generated Python protobuf outputs"
docker run --rm \
  --entrypoint /bin/bash \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "${ROOT_DIR}:/workspace/ray" \
  -v "${UV_CACHE_HOST}:/mnt/shared/uv_cache" \
  -v "${BAZELISK_CACHE_HOST}:/mnt/shared/bazelisk" \
  -w /workspace/ray \
  "${IMAGE}" \
  -lc "
    set -euo pipefail
    chown -R \"\${HOST_UID}:\${HOST_GID}\" python/ray/core/generated python/ray/serve/generated
    mkdir -p '${UV_CACHE_INNER}'
    chown -R \"\${HOST_UID}:\${HOST_GID}\" '${UV_CACHE_INNER}'
    mkdir -p '${BAZELISK_HOME_INNER}'
    chown -R \"\${HOST_UID}:\${HOST_GID}\" '${BAZELISK_HOME_INNER}'
  "

echo "[1/4] Build ray_pkg in Zephyr container"
docker run --rm \
  --entrypoint /bin/bash \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e BAZELISK_HOME="${BAZELISK_HOME_INNER}" \
  -v "${ROOT_DIR}:/workspace/ray" \
  -v "${BAZEL_CACHE_HOST}:/mnt/shared/bazel_cache" \
  -v "${BAZELISK_CACHE_HOST}:/mnt/shared/bazelisk" \
  -v "${UV_CACHE_HOST}:/mnt/shared/uv_cache" \
  -w /workspace/ray \
  "${IMAGE}" \
  -lc "
    set -euo pipefail
    source /opt/spack_src/share/spack/setup-env.sh
    spack env activate /opt/spack_env/default >/dev/null
    bazel --batch --output_user_root='${BAZEL_OUTPUT_ROOT}' build //:ray_pkg --verbose_failures
  "

echo "[2/4] Run focused post-build tests"
docker run --rm \
  --entrypoint /bin/bash \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e BAZELISK_HOME="${BAZELISK_HOME_INNER}" \
  -v "${ROOT_DIR}:/workspace/ray" \
  -v "${BAZEL_CACHE_HOST}:/mnt/shared/bazel_cache" \
  -v "${BAZELISK_CACHE_HOST}:/mnt/shared/bazelisk" \
  -w /workspace/ray \
  "${IMAGE}" \
  -lc "
    set -euo pipefail
    source /opt/spack_src/share/spack/setup-env.sh
    spack env activate /opt/spack_env/default >/dev/null
    bazel --batch --output_user_root='${BAZEL_OUTPUT_ROOT}' test \
      --test_output=errors \
      //src/ray/common/tests:source_location_test \
      //src/ray/common/tests:status_or_test \
      //src/ray/raylet/tests:wait_manager_test
  "

echo "[3/4] Create fresh uv env and install Ray from freshly built tree"
docker run --rm \
  --entrypoint /bin/bash \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e BAZELISK_HOME="${BAZELISK_HOME_INNER}" \
  -v "${ROOT_DIR}:/workspace/ray" \
  -v "${UV_CACHE_HOST}:/mnt/shared/uv_cache" \
  -v "${BAZELISK_CACHE_HOST}:/mnt/shared/bazelisk" \
  -w /workspace/ray \
  "${IMAGE}" \
  -lc "
    set -euo pipefail
    source /opt/spack_src/share/spack/setup-env.sh
    spack env activate /opt/spack_env/default >/dev/null

    export UV_CACHE_DIR='${UV_CACHE_INNER}'
    SPACK_PYTHON=\"\$(command -v python3)\"
    PY_INCLUDES=\"\$(\"\${SPACK_PYTHON}-config\" --includes 2>/dev/null || true)\"
    SPACK_PY_MM=\"\$(\"\${SPACK_PYTHON}\" -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')\"
    if [[ \"\${SPACK_PY_MM}\" != \"${PYTHON_VERSION}\" ]]; then
      echo \"ERROR: Spack Python version \${SPACK_PY_MM} does not match required ${PYTHON_VERSION}\" >&2
      exit 1
    fi

    # Ensure clean packaging scratch owned by current user.
    rm -rf python/build python/ray.egg-info

    # uv environment must use Spack-baked Python and see system packages.
    uv venv --python \"\${SPACK_PYTHON}\" --system-site-packages .venv-ray-pkg-py313
    source .venv-ray-pkg-py313/bin/activate

    cat > '${RAY_EXCLUDES_FILE}' <<'EXCLUDES'
torch
jax
jaxlib
triton
EXCLUDES

    # Reuse freshly built binaries already materialized by //:ray_pkg.
    # Keep Python build headers sourced from Spack Python.
    export CFLAGS=\"\${PY_INCLUDES} \${CFLAGS:-}\"
    export CPPFLAGS=\"\${PY_INCLUDES} \${CPPFLAGS:-}\"

    SKIP_BAZEL_BUILD=1 RAY_BUILD_REDIS=0 uv pip install \
      --python \"\$(pwd)/.venv-ray-pkg-py313/bin/python\" \
      --excludes '${RAY_EXCLUDES_FILE}' \
      ./python

    echo \"Spack Python: \${SPACK_PYTHON}\"
    \"\$(pwd)/.venv-ray-pkg-py313/bin/python\" -c 'import sys; print(\"venv_python=\", sys.executable); print(\"venv_version=\", sys.version)'
  "

echo "[4/4] Run real Ray distributed workload"
docker run --rm \
  --entrypoint /bin/bash \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e BAZELISK_HOME="${BAZELISK_HOME_INNER}" \
  -v "${ROOT_DIR}:/workspace/ray" \
  -v "${BAZELISK_CACHE_HOST}:/mnt/shared/bazelisk" \
  -w /workspace/ray \
  "${IMAGE}" \
  -lc "
    set -euo pipefail
    source .venv-ray-pkg-py313/bin/activate
    python - <<'PY'
import time
import ray

ray.init(num_cpus=4, include_dashboard=False, log_to_driver=True)

@ray.remote
def count_primes(limit: int) -> int:
    count = 0
    for n in range(2, limit):
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            count += 1
    return count

@ray.remote
class Accumulator:
    def __init__(self):
        self.total = 0
    def add(self, value: int):
        self.total += value
        return self.total
    def get(self):
        return self.total

start = time.time()
limits = [20000 + i * 250 for i in range(24)]
counts = ray.get([count_primes.remote(limit) for limit in limits])

actor = Accumulator.remote()
_ = ray.get([actor.add.remote(v) for v in counts])
final_total = ray.get(actor.get.remote())

payload_len = 32 * 1024 * 1024
obj = ray.put(b'x' * payload_len)
roundtrip_len = len(ray.get(obj))

assert len(ray.nodes()) >= 1
assert final_total == sum(counts)
assert roundtrip_len == payload_len

print('ray_version=', ray.__version__)
print('nodes=', len(ray.nodes()))
print('tasks=', len(counts))
print('sum_counts=', final_total)
print('object_roundtrip_bytes=', roundtrip_len)
print('elapsed_sec=', round(time.time() - start, 3))

ray.shutdown()
PY
  "

echo "E2E validation completed successfully."
