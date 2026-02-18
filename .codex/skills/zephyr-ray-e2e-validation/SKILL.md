---
name: zephyr-ray-e2e-validation
description: Run comprehensive end-to-end Ray validation in the Zephyr sglang container using Spack+uv layering, including native bazel build/test, fresh uv install, and real runtime workload checks. This skill is scoped to the Ray repo at /mnt/data_infra/workspace/ray.
---

# Zephyr Ray E2E Validation

## Overview

Use this skill to execute a deterministic Ray validation workflow in Zephyr container infra:

1. Start Zephyr compose service with GPU/mounts/network.
2. Validate Spack Python + GPU visibility.
3. Run native Bazel build and focused C++ tests.
4. Create a fresh uv env from Spack Python and install local Ray with excludes.
5. Run real workload checks (tasks, actor, placement group, object roundtrip).
6. Emit pass/fail summary and preserve logs.

This skill is strictly scoped to the Ray repo at `/mnt/data_infra/workspace/ray`.

## Prerequisites

- Docker daemon running with NVIDIA runtime support.
- Ray repo includes:
  - `dev/zephyr/docker-compose.zephyr-ray.yml`
  - `dev/zephyr/enter_zephyr_ray.sh`
  - `dev/zephyr_ray_e2e_validate.sh`
- Shared infra directories exist and are writable:
  - `/mnt/data_infra/zephyr_container_infra/shared/bazel_cache`
  - `/mnt/data_infra/zephyr_container_infra/shared/uv_cache`
  - `/mnt/data_infra/zephyr_container_infra/shared/hf_cache`
  - `/mnt/data_infra/zephyr_container_infra/shared/hf_cache/hub`

## Quick Start

```bash
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
SKILL_DIR="$CODEX_HOME/skills/zephyr-ray-e2e-validation"

"$SKILL_DIR/scripts/run_zephyr_ray_e2e.sh"
```

## Workflow

### 1) Standard comprehensive run (recommended)

```bash
"$SKILL_DIR/scripts/run_zephyr_ray_e2e.sh"
```

What it does:

- Brings up compose service (`zephyr-ray`)
- Checks GPU visibility and Spack Python activation
- Executes `dev/zephyr_ray_e2e_validate.sh`
- Runs expanded Bazel test suite (16 tests)
- Builds fresh uv env and runs runtime workload checks
- Stops compose service on exit

### 2) Keep container up for debugging

```bash
"$SKILL_DIR/scripts/run_zephyr_ray_e2e.sh" --no-down
```

Then inspect:

```bash
WORKSPACE_DIR=/mnt/data_infra/workspace/ray \
  docker compose -f /mnt/data_infra/workspace/ray/dev/zephyr/docker-compose.zephyr-ray.yml logs -f zephyr-ray
```

The script does not accept custom workspace paths and hard-fails if
`/mnt/data_infra/workspace/ray` is not a Ray repo checkout.

## Expected Success Signals

- Compose service starts and sees GPUs (`nvidia-smi -L` output).
- `torch.cuda.is_available()` is `True` (under Spack env).
- `jax.devices()` lists CUDA devices.
- Bazel expanded suite passes:
  - `Executed 0 out of 16 tests: 16 tests pass.` (cached is acceptable)
- Runtime workload prints:
  - `ray_version ...`
  - `nodes 1+`
  - `task_checksums ...`
  - `actor_total 5050`
  - `object_bytes 67108864`

## Known Non-Blocking Warnings

- Dashboard startup can fail (`return code -11`) in this environment.
  - This is tracked as TODO and is non-blocking for core runtime validation.
- uv may warn about hardlink fallback.
  - This is non-blocking.

## Failure Triage

1. **Permission denied on generated protobuf cleanup**
   - Re-run `dev/zephyr_ray_e2e_validate.sh` which normalizes ownership in stage `[0/4]`.
2. **uv cache permission errors**
   - Verify write access to `/mnt/data_infra/zephyr_container_infra/shared/uv_cache`.
3. **Ray import failure after install**
   - Capture traceback from skill logs.
   - Re-run with `--no-down` and inspect in-container site-packages + versions.
4. **`ray start --head` with client port fails**
   - Minimal install may not include `ray[client]`; this does not invalidate core task/actor checks.

## Script Resource

- `scripts/run_zephyr_ray_e2e.sh`: single-command execution wrapper for this full validation workflow.
