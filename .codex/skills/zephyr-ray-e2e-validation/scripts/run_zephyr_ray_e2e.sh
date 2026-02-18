#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="/mnt/data_infra/workspace/ray"
NO_DOWN=0

usage() {
  cat <<'USAGE'
Usage:
  run_zephyr_ray_e2e.sh [--no-down]

Options:
  --no-down           Keep compose service up after run
  -h, --help          Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-down)
      NO_DOWN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "ERROR: workspace not found: $WORKSPACE_DIR" >&2
  exit 1
fi
if [[ ! -d "$WORKSPACE_DIR/.git" || ! -f "$WORKSPACE_DIR/WORKSPACE" ]]; then
  echo "ERROR: expected Ray repo at $WORKSPACE_DIR (missing .git or WORKSPACE)" >&2
  exit 1
fi

COMPOSE_FILE="$WORKSPACE_DIR/dev/zephyr/docker-compose.zephyr-ray.yml"
E2E_SCRIPT="$WORKSPACE_DIR/dev/zephyr_ray_e2e_validate.sh"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "ERROR: compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi
if [[ ! -x "$E2E_SCRIPT" ]]; then
  echo "ERROR: e2e script not executable/not found: $E2E_SCRIPT" >&2
  exit 1
fi

for d in \
  /mnt/data_infra/zephyr_container_infra/shared/bazel_cache \
  /mnt/data_infra/zephyr_container_infra/shared/bazelisk_cache \
  /mnt/data_infra/zephyr_container_infra/shared/uv_cache \
  /mnt/data_infra/zephyr_container_infra/shared/hf_cache \
  /mnt/data_infra/zephyr_container_infra/shared/hf_cache/hub; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: required host directory missing: $d" >&2
    exit 1
  fi
done

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="$WORKSPACE_DIR/.zephyr-e2e-logs"
mkdir -p "$log_dir"
log_file="$log_dir/e2e-$timestamp.log"

compose() {
  WORKSPACE_DIR="$WORKSPACE_DIR" docker compose -f "$COMPOSE_FILE" "$@"
}

cleanup() {
  if [[ "$NO_DOWN" -eq 0 ]]; then
    compose down --remove-orphans >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

{
  echo "== Zephyr Ray E2E =="
  echo "workspace=$WORKSPACE_DIR"
  echo "log_file=$log_file"
  echo

  echo "[A] Compose up"
  compose up -d zephyr-ray
  compose ps
  echo

  echo "[B] GPU + Spack sanity"
  compose exec -T zephyr-ray bash -lc '
    set -euo pipefail
    nvidia-smi -L | head -n 8
    export HOME=/tmp
    source /opt/spack_src/share/spack/setup-env.sh
    spack env activate /opt/spack_env/default >/dev/null
    python3 - <<'"'"'PY'"'"'
import sys
import torch
import jax
print("python", sys.version.split()[0])
print("torch_cuda", torch.cuda.is_available())
print("jax_devices", jax.devices())
PY
  '
  echo

  echo "[C] Native e2e script"
  BAZEL_OUTPUT_ROOT=/workspace/ray/.bazel-output-root "$E2E_SCRIPT"
  echo

  echo "[D] Expanded Bazel tests"
  compose exec -T zephyr-ray bash -lc '
    set -euo pipefail
    export HOME=/tmp
    export XDG_CACHE_HOME=/tmp/.cache
    source /opt/spack_src/share/spack/setup-env.sh
    spack env activate /opt/spack_env/default >/dev/null
    bazel --batch --output_user_root=/mnt/shared/bazel_cache/ray-sglang-codex-build test \
      --test_output=errors --verbose_failures \
      //src/ray/common/tests:all \
      //src/ray/raylet/tests:wait_manager_test \
      //src/ray/raylet/tests:lease_dependency_manager_test \
      //src/ray/gcs/tests:gcs_node_manager_test \
      //src/ray/gcs/tests:gcs_job_manager_test \
      //src/ray/gcs/tests:gcs_task_manager_test
  '
  echo

  echo "[E] Comprehensive runtime workload"
  compose exec -T zephyr-ray bash -lc '
    set -euo pipefail
    export HOME=/tmp
    export XDG_CACHE_HOME=/tmp/.cache
    source /opt/spack_src/share/spack/setup-env.sh
    spack env activate /opt/spack_env/default >/dev/null
    source .venv-ray-pkg-py313/bin/activate
    uv pip install numpy
    python - <<'"'"'PY'"'"'
import time
import ray
from ray.util.placement_group import placement_group

ray.init(num_cpus=4, include_dashboard=False, log_to_driver=True)

@ray.remote
def matmul_checksum(n: int, seed: int) -> float:
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)).astype("float32")
    b = rng.normal(size=(n, n)).astype("float32")
    return float((a @ b).sum())

@ray.remote
class Counter:
    def __init__(self):
        self.v = 0
    def add(self, x: int):
        self.v += x
        return self.v
    def get(self):
        return self.v

start = time.time()
checksums = ray.get([matmul_checksum.remote(256, i) for i in range(8)])
ctr = Counter.remote()
ray.get([ctr.add.remote(i) for i in range(1, 101)])
actor_total = ray.get(ctr.get.remote())

pg = placement_group([{"CPU": 1}, {"CPU": 1}], strategy="PACK")
ray.get(pg.ready())

@ray.remote(num_cpus=1)
def pg_task(x):
    return x * x

s = ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy
r1 = pg_task.options(scheduling_strategy=s(placement_group=pg, placement_group_bundle_index=0)).remote(7)
r2 = pg_task.options(scheduling_strategy=s(placement_group=pg, placement_group_bundle_index=1)).remote(9)
vals = ray.get([r1, r2])

payload = b"y" * (64 * 1024 * 1024)
out = ray.get(ray.put(payload))

assert len(checksums) == 8
assert actor_total == 5050
assert vals == [49, 81]
assert len(out) == len(payload)

print("ray_version", ray.__version__)
print("nodes", len(ray.nodes()))
print("task_checksums", len(checksums))
print("actor_total", actor_total)
print("pg_vals", vals)
print("object_bytes", len(out))
print("elapsed_sec", round(time.time() - start, 3))

ray.shutdown()
PY
  '
  echo
  echo "PASS: comprehensive Zephyr Ray E2E validation completed"
} 2>&1 | tee "$log_file"

echo "Logs written to: $log_file"
