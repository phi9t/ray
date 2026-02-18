# AGENTS.md

## What is Ray?

Ray is a unified framework for scaling AI and Python applications. It provides core distributed primitives (tasks, actors, objects) and higher-level AI libraries (Data, Train, Tune, RLlib, Serve) built on top of a C++ distributed runtime.

## Build System

Ray uses **Bazel 6.5.0** for the C++ core and **setuptools** for the Python package.

```bash
# Build from source (Python + C++ core)
cd python && uv pip install --system -e . --verbose

# Skip C++ rebuild (Python-only changes)
SKIP_BAZEL_BUILD=1 uv pip install --system -e . --verbose

# Build C++ core only
bazel build //:ray_pkg

# Build with debug symbols
RAY_DEBUG_BUILD=1 uv pip install --system -e . --verbose
```

Key environment variables:
- `RAY_BUILD_CORE=1` — build C++ core (default on)
- `SKIP_BAZEL_BUILD=1` — skip Bazel entirely for Python-only changes
- `RAY_INSTALL_JAVA=1` — also build Java bindings
- `BAZEL_LIMIT_CPUS` — limit Bazel CPU usage

## Running Tests

Tests use **pytest** with a default 180-second timeout per test (configured in `pytest.ini`).

```bash
# Run a single test file
pytest python/ray/tests/test_basic.py

# Run a specific test
pytest python/ray/tests/test_basic.py::test_function_name

# Run with custom timeout
pytest --timeout=300 python/ray/tests/test_basic.py

# Run C++ tests via Bazel
bazel test //src/ray/core_worker:core_worker_test

# Run Bazel test with CI config
bazel test --config=ci //src/ray/gcs/...
```

Tests for each library live alongside the code:
- Core: `python/ray/tests/`
- Data: `python/ray/data/tests/`
- Serve: `python/ray/serve/tests/`
- Train: `python/ray/train/tests/`
- Tune: `python/ray/tune/tests/`
- RLlib: `rllib/tests/` and within `rllib/algorithms/*/tests/`

## Linting and Formatting

Python formatting uses **black** (line-length 88) and **ruff** for linting/import sorting. C++ uses **clang-format** (v12). Bazel BUILD files use **buildifier**.

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run clang-format --all-files

# Run the CI lint script (comprehensive)
ci/lint/lint.sh pre_commit
```

Ruff config is in `pyproject.toml` — extends with rules I (isort), B (bugbear), Q (quotes), C4 (comprehensions), W (whitespace). Import ordering has a custom `afterray` section for `psutil` and `setproctitle`.

## Architecture

### Language Layers

- **C++ core** (`src/ray/`): The distributed runtime — GCS, raylet, object manager, core worker, RPC, pub/sub. This is compiled via Bazel and linked through Cython bindings.
- **Cython bridge** (`python/ray/_raylet.pyx`): Binds C++ core to Python. Compiles to `_raylet.so`.
- **Python API** (`python/ray/`): User-facing APIs and library implementations.

### C++ Core Components (`src/ray/`)

| Directory | Purpose |
|-----------|---------|
| `gcs/` | Global Control Store — cluster metadata, actor registry, resource management |
| `raylet/` | Per-node scheduler — task dispatch, local resource management |
| `core_worker/` | Worker process runtime — task execution, object handling, Cython interface |
| `object_manager/` | Distributed object transfer between nodes |
| `rpc/` | gRPC-based inter-process communication |
| `pubsub/` | Internal pub/sub messaging system |
| `common/` | Shared utilities — scheduling, cgroup2, syncer |
| `protobuf/` | Protocol buffer definitions for all IPC |

### Python Package Structure (`python/ray/`)

| Directory | Purpose |
|-----------|---------|
| `_private/` | Internal implementation (worker, runtime env, logging, telemetry) |
| `_common/` | Shared utilities across Ray components |
| `dag/` | DAG compilation for aDAG (accelerated DAG) execution |
| `data/` | Ray Data — scalable data processing (Dataset API) |
| `serve/` | Ray Serve — model serving with batching, deployments |
| `train/` | Ray Train — distributed training (v1 and v2 APIs coexist) |
| `tune/` | Ray Tune — hyperparameter tuning |
| `air/` | Ray AIR — unified ML API layer |
| `autoscaler/` | Cluster autoscaling logic |
| `dashboard/` | Web dashboard backend (React/TypeScript frontend in `dashboard/client/`) |
| `runtime_env/` | Runtime environment management (pip, conda, containers) |
| `llm/` | LLM-specific serving utilities |
| `core/generated/` | Auto-generated protobuf Python bindings (do not edit) |

### RLlib (`rllib/`)

Reinforcement learning library with its own directory structure:
- `algorithms/` — RL algorithm implementations (PPO, DQN, etc.)
- `core/` — Core RL abstractions
- `connectors/` — Data pipeline connectors
- `env/` — Environment wrappers
- `models/` — Neural network model definitions

### Key Patterns

- **Generated code**: `python/ray/core/generated/` and `python/ray/serve/generated/` contain protobuf-generated files — never edit these directly. Regenerate with `gen_py_proto.py`.
- **Thirdparty files**: `python/ray/thirdparty_files/` and `python/ray/_private/thirdparty/` are vendored — excluded from linting.
- **Ray Train v2**: A v2 API exists alongside v1 in `python/ray/train/v2/`. There's a circular import lint check specifically for train (`check_circular_imports.py`).
- **Actor/Task lifecycle**: Design docs in `src/ray/design_docs/` describe actor states, task states, and ID specifications.


## Installation Command Policy

All installation commands in this repository must use `uv`.

- Do not use `pip install`, `pip3 install`, or `python -m pip install` in scripts, docs, comments, or examples.
- Use `uv pip install` instead.
- In non-venv/global contexts, use `uv pip install --system`.

## Required Conversions

- `pip install <pkg>` -> `uv pip install --system <pkg>`
- `pip3 install <pkg>` -> `uv pip install --system <pkg>`
- `python -m pip install <args>` -> `uv pip install --system <args>`
- `python3 -m pip install <args>` -> `uv pip install --system <args>`

## Compatibility Exception

Do not rename stable API/config field names that include `pip` (for example
`pip_check`, `pip_install_options`) if they are part of public interfaces.
Only update command guidance and examples around them to use `uv`.

## Zephyr Build/Run Guide

For the canonical Zephyr container workflow (Spack + uv layering) to build and
run Ray, see:

- `foundation.org`
- `doc/zephyr_spack_uv_ray_build_enablement.org`

Canonical command surface (hard-break) is:

- `dev/zephyr/zephyr_ray.sh`
- `just` targets are thin aliases to this orchestrator.
- Release gate command: `dev/zephyr/zephyr_ray.sh release-ready` (or `just release`)
- Release gate is strict: it hard-fails if `torch`/`jax` are unavailable in Ray runtime context, if they do not resolve from Spack paths, or if excluded deps are installed inside the Ray venv.
