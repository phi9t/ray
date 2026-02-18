#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.zephyr-ray.yml"
SERVICE_NAME="zephyr-ray"

WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}}"
CONTAINER_HOME_BASE="${CONTAINER_HOME_BASE:-/mnt/data_infra/zephyr_container_infra/shared/homes/ray}"
CONTAINER_HOME_HOST="${CONTAINER_HOME_HOST:-${CONTAINER_HOME_BASE}-$(id -u)}"
CONTAINER_HOME_INNER="${CONTAINER_HOME_INNER:-/home/ray}"
SHARED_BAZEL_CACHE_DIR="${SHARED_BAZEL_CACHE_DIR:-/mnt/data_infra/zephyr_container_infra/shared/bazel_cache}"
SHARED_BAZELISK_CACHE_DIR="${SHARED_BAZELISK_CACHE_DIR:-/mnt/data_infra/zephyr_container_infra/shared/bazelisk_cache}"
SHARED_UV_CACHE_DIR="${SHARED_UV_CACHE_DIR:-/mnt/data_infra/zephyr_container_infra/shared/uv_cache}"
SHARED_HF_CACHE_DIR="${SHARED_HF_CACHE_DIR:-/mnt/data_infra/zephyr_container_infra/shared/hf_cache}"
SHARED_HF_MODELS_DIR="${SHARED_HF_MODELS_DIR:-/mnt/data_infra/zephyr_container_infra/shared/hf_cache/hub}"
BAZEL_OUTPUT_ROOT="${BAZEL_OUTPUT_ROOT:-/workspace/ray/.bazel-output-root}"
BAZEL_TEST_OUTPUT_ROOT="${BAZEL_TEST_OUTPUT_ROOT:-/workspace/ray/.bazel-output-root}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
RAY_VENV_DIR="${RAY_VENV_DIR:-/workspace/ray/.venv-ray-pkg-py313}"
RAY_EXCLUDES_FILE_HOST="${RAY_EXCLUDES_FILE_HOST:-${WORKSPACE_DIR}/dev/zephyr/ray_uv_excludes.txt}"
RAY_EXCLUDES_FILE_INNER="${RAY_EXCLUDES_FILE_INNER:-/workspace/ray/dev/zephyr/ray_uv_excludes.txt}"
UV_CACHE_SUBDIR="${UV_CACHE_SUBDIR:-ray-install}"
UV_CACHE_DIR_INNER="${UV_CACHE_DIR_INNER:-/mnt/shared/uv_cache/${UV_CACHE_SUBDIR}}"
SPACK_ENV_PATH="${SPACK_ENV_PATH:-/opt/spack_env/default}"
SPACK_SETUP_PATH="${SPACK_SETUP_PATH:-/opt/spack_src/share/spack/setup-env.sh}"
BAZEL_ACTION_PATH="${BAZEL_ACTION_PATH:-/opt/spack_store/view/bin:/usr/local/bin:/usr/bin:/bin}"
RELEASE_LOG_DIR="${RELEASE_LOG_DIR:-${WORKSPACE_DIR}/.zephyr-e2e-logs}"
REQUIRED_EXCLUDE_DEPS=("torch" "jax" "jaxlib" "triton")

log() {
  printf '[zephyr-ray] %s\n' "$*"
}

die() {
  printf '[zephyr-ray] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage:
  dev/zephyr/zephyr_ray.sh up
  dev/zephyr/zephyr_ray.sh down
  dev/zephyr/zephyr_ray.sh status
  dev/zephyr/zephyr_ray.sh logs
  dev/zephyr/zephyr_ray.sh shell
  dev/zephyr/zephyr_ray.sh preflight
  dev/zephyr/zephyr_ray.sh debug-env
  dev/zephyr/zephyr_ray.sh build
  dev/zephyr/zephyr_ray.sh install [--venv-reuse|--venv-clear]
  dev/zephyr/zephyr_ray.sh run-smoke
  dev/zephyr/zephyr_ray.sh run-e2e
  dev/zephyr/zephyr_ray.sh run-e2e-comprehensive
  dev/zephyr/zephyr_ray.sh release-ready
USAGE
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_dir_rw() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || die "${label} directory does not exist: ${path}"
  [[ -r "${path}" && -w "${path}" ]] || die "${label} directory is not read/write: ${path}"
}

compose() {
  WORKSPACE_DIR="${WORKSPACE_DIR}" \
  CONTAINER_HOME_HOST="${CONTAINER_HOME_HOST}" \
  CONTAINER_HOME_INNER="${CONTAINER_HOME_INNER}" \
  SHARED_BAZEL_CACHE_DIR="${SHARED_BAZEL_CACHE_DIR}" \
  SHARED_BAZELISK_CACHE_DIR="${SHARED_BAZELISK_CACHE_DIR}" \
  SHARED_UV_CACHE_DIR="${SHARED_UV_CACHE_DIR}" \
  SHARED_HF_CACHE_DIR="${SHARED_HF_CACHE_DIR}" \
  SHARED_HF_MODELS_DIR="${SHARED_HF_MODELS_DIR}" \
  BAZEL_OUTPUT_ROOT="${BAZEL_OUTPUT_ROOT}" \
  docker compose -f "${COMPOSE_FILE}" "$@"
}

check_prereqs() {
  require_cmd docker
  docker compose version >/dev/null 2>&1 || die "docker compose plugin is required"
  docker info >/dev/null 2>&1 || die "docker daemon is not reachable"
  [[ -f "${WORKSPACE_DIR}/WORKSPACE" ]] || die "Ray WORKSPACE file missing at ${WORKSPACE_DIR}"
  [[ -f "${COMPOSE_FILE}" ]] || die "compose file missing at ${COMPOSE_FILE}"
  require_dir_rw "${WORKSPACE_DIR}" "workspace"
  mkdir -p "${CONTAINER_HOME_HOST}"
  require_dir_rw "${CONTAINER_HOME_HOST}" "container-scoped home"
  require_dir_rw "${SHARED_BAZEL_CACHE_DIR}" "shared bazel cache"
  require_dir_rw "${SHARED_BAZELISK_CACHE_DIR}" "shared bazelisk cache"
  require_dir_rw "${SHARED_UV_CACHE_DIR}" "shared uv cache"
  require_dir_rw "${SHARED_HF_CACHE_DIR}" "shared hf cache"
  require_dir_rw "${SHARED_HF_MODELS_DIR}" "shared hf models"
  [[ -f "${RAY_EXCLUDES_FILE_HOST}" ]] || die "missing Ray excludes file: ${RAY_EXCLUDES_FILE_HOST}"
  compose config -q >/dev/null 2>&1 || die "compose file/env resolution failed; run: docker compose -f ${COMPOSE_FILE} config"
}

validate_excludes_file() {
  local dep
  for dep in "${REQUIRED_EXCLUDE_DEPS[@]}"; do
    if ! rg -n -x "${dep}" "${RAY_EXCLUDES_FILE_HOST}" >/dev/null 2>&1; then
      die "Ray excludes file is missing required dependency '${dep}': ${RAY_EXCLUDES_FILE_HOST}"
    fi
  done
}

verify_spack_gpu_runtime() {
  up
  log "verifying Spack Python/toolchain and GPU runtime"
  run_user_spack "set -euo pipefail; \
    command -v nvidia-smi >/dev/null; \
    nvidia-smi -L >/dev/null; \
    python3 - <<'PY'
import os
import platform
import subprocess
import sys
import torch
import jax
print('python=', sys.executable)
print('python_version=', platform.python_version())
print('torch_version=', torch.__version__)
print('torch_cuda_available=', torch.cuda.is_available())
print('jax_version=', jax.__version__)
print('jax_devices=', [str(d) for d in jax.devices()])
if not torch.cuda.is_available():
    raise SystemExit('torch.cuda.is_available() is False; GPU runtime is not healthy')
subprocess.run(['nvidia-smi', '-L'], check=True)
print('home=', os.environ.get('HOME'))
print('spack_user_config=', os.environ.get('SPACK_USER_CONFIG_PATH'))
PY"
}

verify_uv_layering() {
  up
  log "verifying uv layered install uses Spack torch/jax and does not override excluded deps"
  run_user_spack_venv "set -euo pipefail; \
    python - <<'PY'
import importlib
from importlib import metadata as md
from pathlib import Path
import os
venv = Path('${RAY_VENV_DIR}').resolve()
required = {'torch', 'jax'}
spack_roots = [Path(p).resolve() for p in os.environ.get('SPACK_SITE_PATHS', '').split(':') if p]
allowed_required_prefixes = ('/opt/spack_store/',)
bad = []
for name in ('torch', 'jax', 'jaxlib', 'triton'):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f'{name}_module=absent ({exc})')
        if name in required:
            bad.append(f'{name} import failed in Ray runtime context: {exc}')
        continue
    mod_path = getattr(module, '__file__', None)
    print(f'{name}_module={mod_path}')
    if mod_path is None:
        bad.append(f'{name} module has no __file__; cannot verify source location')
    else:
        mod_resolved = Path(mod_path).resolve()
        if str(mod_resolved).startswith(str(venv)):
            bad.append(f'{name} module loaded from Ray venv: {mod_resolved}')
        if name in required:
            if not str(mod_resolved).startswith(allowed_required_prefixes):
                bad.append(f'{name} module is not from Spack paths: {mod_resolved}')
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        print(f'{name}_dist=absent')
        if name in required:
            bad.append(f'{name} distribution metadata missing in runtime context')
        continue
    loc = Path(dist.locate_file('')).resolve()
    print(f'{name}_dist={loc}')
    if str(loc).startswith(str(venv)):
        bad.append(f'{name} installed inside Ray venv: {loc}')
if spack_roots:
    print('spack_site_paths=', ':'.join(str(p) for p in spack_roots))
else:
    bad.append('SPACK_SITE_PATHS is empty; Spack environment discovery failed')
if bad:
    raise SystemExit('\\n'.join(bad))
print('uv_layering=ok')
PY"
}

write_release_bom() {
  up
  log "capturing release bill of materials"
  run_user_spack "set -euo pipefail; \
    echo '== excludes =='; \
    python - <<'PY'
import hashlib
from pathlib import Path
p = Path('${RAY_EXCLUDES_FILE_INNER}')
data = p.read_bytes()
print('excludes_file=', p)
print('excludes_sha256=', hashlib.sha256(data).hexdigest())
print('excludes_contents=')
for line in data.decode('utf-8').splitlines():
    print(line)
PY
    SPACK_PYTHON=\$(command -v python3); \
    SPACK_SITE_PATHS=\$(\"\${SPACK_PYTHON}\" - <<'PY'
import site
import sysconfig
paths = []
for p in site.getsitepackages():
    if p and p not in paths:
        paths.append(p)
for key in ('purelib', 'platlib'):
    p = sysconfig.get_paths().get(key)
    if p and p not in paths:
        paths.append(p)
print(':'.join(paths))
PY
); \
    echo '== runtime_context_before_venv =='; \
    \"\${SPACK_PYTHON}\" - <<'PY'
import os
import site
import sys
print('python=', sys.executable)
print('pythonpath=', os.environ.get('PYTHONPATH', ''))
print('sys_path_head=', sys.path[:8])
print('site_packages=', site.getsitepackages())
PY
    BASE_PYTHONPATH=\"\${PYTHONPATH:-}\"; \
    source '${RAY_VENV_DIR}/bin/activate'; \
    export SPACK_SITE_PATHS=\"\${SPACK_SITE_PATHS}\"; \
    if [[ -n \"\${SPACK_SITE_PATHS}\" ]]; then \
      if [[ -n \"\${PYTHONPATH:-}\" ]]; then \
        export PYTHONPATH=\"\${SPACK_SITE_PATHS}:\${PYTHONPATH}\"; \
      else \
        export PYTHONPATH=\"\${SPACK_SITE_PATHS}\"; \
      fi; \
    fi; \
    echo '== runtime_context_after_venv_spack_overlay =='; \
    python - <<'PY'
import os
import sys
import torch
import jax
print('python=', sys.executable)
print('pythonpath=', os.environ.get('PYTHONPATH', ''))
print('sys_path_head=', sys.path[:8])
print('torch_path=', getattr(torch, '__file__', ''))
print('jax_path=', getattr(jax, '__file__', ''))
PY
    echo '== runtime_context_after_venv_clean =='; \
    PYTHONPATH=\"\${BASE_PYTHONPATH}\" python - <<'PY'
import os
import pathlib
import sys
import ray
print('python=', sys.executable)
print('pythonpath=', os.environ.get('PYTHONPATH', ''))
print('sys_path_head=', sys.path[:8])
print('ray_version=', ray.__version__)
print('ray_path=', getattr(ray, '__file__', ''))
print('raylet_path=', pathlib.Path(ray.__file__).with_name('_raylet.so'))
PY
    echo '== versions =='; \
    python -V; \
    uv --version; \
    bazel --version; \
    gcc --version | head -n 1; \
    ld.lld --version | head -n 1"
}

release_ready() {
  check_prereqs
  validate_excludes_file
  mkdir -p "${RELEASE_LOG_DIR}"
  local ts
  local log_file
  local log_abs
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  log_file="${RELEASE_LOG_DIR}/release-ready-${ts}.log"
  {
    log "release-ready gate started (utc=${ts})"
    verify_spack_gpu_runtime
    run_e2e_comprehensive
    verify_uv_layering
    write_release_bom
    log "RELEASE_READY=1"
  } | tee "${log_file}"
  if command -v realpath >/dev/null 2>&1; then
    log_abs="$(realpath "${log_file}")"
  else
    log_abs="${log_file}"
  fi
  log "release log written: ${log_abs}"
}

up() {
  check_prereqs
  log "starting ${SERVICE_NAME}"
  compose up -d "${SERVICE_NAME}"
}

down() {
  check_prereqs
  log "stopping ${SERVICE_NAME}"
  compose down --remove-orphans
}

status() {
  check_prereqs
  compose ps
}

logs() {
  check_prereqs
  compose logs -f --tail=200 "${SERVICE_NAME}"
}

run_root() {
  local cmd="$1"
  compose exec -T \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${SERVICE_NAME}" bash -lc "${cmd}"
}

run_user() {
  local cmd="$1"
  compose exec -T -u "$(id -u):$(id -g)" "${SERVICE_NAME}" bash -lc "${cmd}"
}

run_user_spack() {
  local cmd="$1"
  run_user "set -euo pipefail; \
    export SPACK_DISABLE_LOCAL_CONFIG=1; \
    export SPACK_USER_CONFIG_PATH='${CONTAINER_HOME_INNER}/.spack-clean'; \
    mkdir -p \"\${SPACK_USER_CONFIG_PATH}\" '${CONTAINER_HOME_INNER}/.cache'; \
    export PATH=/opt/spack_store/view/bin:\${PATH}; \
    source '${SPACK_SETUP_PATH}'; \
    spack_activate_out='${CONTAINER_HOME_INNER}/.cache/spack_activate.out'; \
    spack_activate_err='${CONTAINER_HOME_INNER}/.cache/spack_activate.err'; \
    : > \"\${spack_activate_out}\"; \
    : > \"\${spack_activate_err}\"; \
    if ! spack env activate '${SPACK_ENV_PATH}' >\"\${spack_activate_out}\" 2>\"\${spack_activate_err}\"; then \
      cat \"\${spack_activate_err}\" >&2; \
      exit 1; \
    fi; \
    ${cmd}"
}

run_user_spack_venv() {
  local cmd="$1"
  run_user_spack "set -euo pipefail; \
    SPACK_PYTHON=\$(command -v python3); \
    SPACK_SITE_PATHS=\$(\"\${SPACK_PYTHON}\" - <<'PY'
import site
import sysconfig
paths = []
for p in site.getsitepackages():
    if p and p not in paths:
        paths.append(p)
for key in ('purelib', 'platlib'):
    p = sysconfig.get_paths().get(key)
    if p and p not in paths:
        paths.append(p)
print(':'.join(paths))
PY
); \
    source '${RAY_VENV_DIR}/bin/activate'; \
    export SPACK_SITE_PATHS=\"\${SPACK_SITE_PATHS}\"; \
    ${cmd}"
}

shell() {
  up
  log "opening interactive shell"
  local exec_flags=()
  if [[ -t 0 && -t 1 ]]; then
    exec_flags=(-it)
  fi
  compose exec "${exec_flags[@]}" -u "$(id -u):$(id -g)" "${SERVICE_NAME}" bash -lc \
    "set -euo pipefail; \
     export SPACK_DISABLE_LOCAL_CONFIG=1; \
     export SPACK_USER_CONFIG_PATH='${CONTAINER_HOME_INNER}/.spack-clean'; \
     mkdir -p \"\${SPACK_USER_CONFIG_PATH}\" '${CONTAINER_HOME_INNER}/.cache'; \
     export PATH=/opt/spack_store/view/bin:\${PATH}; \
     source '${SPACK_SETUP_PATH}'; \
     spack_activate_out='${CONTAINER_HOME_INNER}/.cache/spack_activate.out'; \
     spack_activate_err='${CONTAINER_HOME_INNER}/.cache/spack_activate.err'; \
     : > \"\${spack_activate_out}\"; \
     : > \"\${spack_activate_err}\"; \
     if ! spack env activate '${SPACK_ENV_PATH}' >\"\${spack_activate_out}\" 2>\"\${spack_activate_err}\"; then \
       cat \"\${spack_activate_err}\" >&2; \
       exit 1; \
     fi; \
     if [[ -d '${RAY_VENV_DIR}' ]]; then \
       SPACK_PYTHON=\$(command -v python3); \
       SPACK_SITE_PATHS=\$(\"\${SPACK_PYTHON}\" - <<'PY'
import site
import sysconfig
paths = []
for p in site.getsitepackages():
    if p and p not in paths:
        paths.append(p)
for key in ('purelib', 'platlib'):
    p = sysconfig.get_paths().get(key)
    if p and p not in paths:
        paths.append(p)
print(':'.join(paths))
PY
); \
       source '${RAY_VENV_DIR}/bin/activate'; \
       export SPACK_SITE_PATHS=\"\${SPACK_SITE_PATHS}\"; \
       echo 'Ray runtime env active. Workspace: /workspace/ray'; \
       echo \"python=\$(command -v python)\"; \
       echo \"ray=\$(command -v ray)\"; \
     else \
       echo 'Spack env active. Workspace: /workspace/ray'; \
       echo 'Ray venv missing; run: just install'; \
     fi; \
     exec bash -i"
}

preflight() {
  up
  log "running preflight checks and ownership normalization"
  # rules_foreign_cc bootstraps GNU make with a sanitized PATH that cannot see
  # the Spack view, so expose ld.lld on a standard system path first.
  run_root "set -euo pipefail; \
    chown -R \${HOST_UID}:\${HOST_GID} /workspace/ray/python/ray/core/generated /workspace/ray/python/ray/serve/generated; \
    mkdir -p '${CONTAINER_HOME_INNER}' '${CONTAINER_HOME_INNER}/.cache' '${CONTAINER_HOME_INNER}/.spack-clean' '${UV_CACHE_DIR_INNER}' '${BAZEL_OUTPUT_ROOT}'; \
    ln -sfn /opt/spack_store/view/bin/ld.lld /usr/local/bin/ld.lld; \
    chown \${HOST_UID}:\${HOST_GID} '${CONTAINER_HOME_INNER}' '${CONTAINER_HOME_INNER}/.cache' '${CONTAINER_HOME_INNER}/.spack-clean'; \
    chown -R \${HOST_UID}:\${HOST_GID} '${UV_CACHE_DIR_INNER}' '${BAZEL_OUTPUT_ROOT}'"
  run_user_spack "command -v ld.lld >/dev/null; command -v gcc >/dev/null; command -v g++ >/dev/null; command -v python3 >/dev/null"
}

debug_env() {
  up
  run_user_spack "python3 - <<'PY'
import os
import platform
import sys
print('python=', sys.executable)
print('python_version=', platform.python_version())
print('home=', os.environ.get('HOME'))
print('spack_user_config=', os.environ.get('SPACK_USER_CONFIG_PATH'))
print('uv_cache=', os.environ.get('UV_CACHE_DIR'))
PY
python3-config --includes
which bazel
which uv
which ld.lld
"
}

build() {
  preflight
  log "building //:ray_pkg"
  run_user_spack "bazel --batch --output_user_root='${BAZEL_OUTPUT_ROOT}' \
    build --action_env=PATH='${BAZEL_ACTION_PATH}' //:ray_pkg --verbose_failures"
}

install() {
  local mode="${1:---venv-reuse}"
  build
  log "installing Ray with uv using Spack Python (${mode})"
  local clear_snippet=""
  if [[ "${mode}" == "--venv-clear" ]]; then
    clear_snippet="rm -rf '${RAY_VENV_DIR}';"
  elif [[ "${mode}" != "--venv-reuse" ]]; then
    die "unknown install mode: ${mode} (expected --venv-reuse or --venv-clear)"
  fi
  run_user_spack "export UV_CACHE_DIR='${UV_CACHE_DIR_INNER}'; \
    export UV_LINK_MODE=copy; \
    SPACK_PYTHON=\$(command -v python3); \
    SPACK_PY_MM=\$(\"\${SPACK_PYTHON}\" -c \"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')\"); \
    if [[ \"\${SPACK_PY_MM}\" != '${PYTHON_VERSION}' ]]; then \
      echo \"ERROR: Spack Python version \${SPACK_PY_MM} does not match required ${PYTHON_VERSION}\" >&2; \
      exit 1; \
    fi; \
    PY_INCLUDES=\$(\"\${SPACK_PYTHON}-config\" --includes 2>/dev/null || true); \
    rm -rf /workspace/ray/python/build /workspace/ray/python/ray.egg-info; \
    ${clear_snippet} \
    if [[ ! -d '${RAY_VENV_DIR}' ]]; then \
      cd /tmp; \
      uv venv --python \"\${SPACK_PYTHON}\" --system-site-packages '${RAY_VENV_DIR}'; \
      cd /workspace/ray; \
    fi; \
    source '${RAY_VENV_DIR}/bin/activate'; \
    CFLAGS=\"\${PY_INCLUDES}\" CPPFLAGS=\"\${PY_INCLUDES}\" SKIP_BAZEL_BUILD=1 RAY_BUILD_REDIS=0 \
      uv pip install --python '${RAY_VENV_DIR}/bin/python' --excludes '${RAY_EXCLUDES_FILE_INNER}' /workspace/ray/python; \
    '${RAY_VENV_DIR}/bin/python' -c \"import ray,sys; print('venv_python=', sys.executable); print('ray_version=', ray.__version__)\""
}

run_smoke() {
  install --venv-reuse
  log "running smoke workload"
  run_user_spack "source '${RAY_VENV_DIR}/bin/activate'; python /workspace/ray/dev/zephyr/ray_job_smoke.py"
}

run_e2e() {
  run_smoke
  log "e2e completed"
}

run_e2e_comprehensive() {
  run_e2e
  log "running focused Bazel tests (toolchain-stable set)"
  run_user_spack "bazel --batch --output_user_root='${BAZEL_TEST_OUTPUT_ROOT}' test --test_output=errors \
    --action_env=PATH='${BAZEL_ACTION_PATH}' \
    //src/ray/common/tests:source_location_test \
    //src/ray/raylet/tests:wait_manager_test"
  if [[ "${RUN_TOOLCHAIN_SENSITIVE_TESTS:-0}" == "1" ]]; then
    log "running toolchain-sensitive Bazel tests (opt-in)"
    run_user_spack "bazel --batch --output_user_root='${BAZEL_TEST_OUTPUT_ROOT}' test --test_output=errors \
      --action_env=PATH='${BAZEL_ACTION_PATH}' \
      //src/ray/common/tests:status_or_test"
  else
    log "skipping toolchain-sensitive tests; set RUN_TOOLCHAIN_SENSITIVE_TESTS=1 to enable //src/ray/common/tests:status_or_test"
  fi
  log "running additional Python integration checks"
  run_user_spack "source '${RAY_VENV_DIR}/bin/activate'; python - <<'PY'
import ray
ray.init(num_cpus=2, include_dashboard=False, log_to_driver=True)
@ray.remote
def f(x):
    return x + 1
@ray.remote
class A:
    def ping(self):
        return 'pong'
vals = ray.get([f.remote(i) for i in range(16)])
assert vals[0] == 1 and vals[-1] == 16
a = A.remote()
assert ray.get(a.ping.remote()) == 'pong'
ray.shutdown()
print('python_integration=ok')
PY"
  log "comprehensive validation completed"
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    up) up ;;
    down) down ;;
    status) status ;;
    logs) logs ;;
    shell|"") shell ;;
    preflight) preflight ;;
    debug-env) debug_env ;;
    build) build ;;
    install) install "${2:---venv-reuse}" ;;
    run-smoke) run_smoke ;;
    run-e2e) run_e2e ;;
    run-e2e-comprehensive) run_e2e_comprehensive ;;
    release-ready) release_ready ;;
    -h|--help|help) usage ;;
    *)
      die "unknown command: ${cmd}"
      ;;
  esac
}

main "$@"
