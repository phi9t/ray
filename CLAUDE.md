# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Ray

Ray is a unified framework for scaling AI and Python applications. It provides core primitives (tasks, actors, objects) and libraries (Data, Train, Tune, RLlib, Serve) for distributed computing.

## Build System

Ray uses **Bazel** (v6.5.0) for C++ and **setuptools** for Python packaging.

```bash
# Build from source (Python)
cd python && pip install -e . --verbose

# Build C++ only
bazel build //:ray_pkg

# Python-only dev (after installing a nightly wheel)
pip install -U <ray-nightly-wheel>
python python/ray/setup-dev.py
```

Key environment variables:
- `RAY_INSTALL_JAVA=1` / `RAY_INSTALL_CPP=1` — include Java/C++ components
- `SKIP_BAZEL_BUILD=1` — skip native compilation
- `RAY_DEBUG_BUILD=debug|asan|tsan` — debug/sanitizer builds

## Testing

Python tests use **pytest** (default timeout: 180s per test):

```bash
pip install -c python/requirements_compiled.txt -r python/requirements/test-requirements.txt

# Run a single test file
python -m pytest -v -s python/ray/tests/test_basic.py

# Run a single test by name
python -m pytest -v -s python/ray/tests/test_basic.py::test_name
```

C++ tests use **Google Test** via Bazel:

```bash
bazel test --config=ci //src/ray/core_worker:core_worker_test
bazel test --config=ci --test_filter=TestName --test_output=streamed <target>
```

## Linting and Formatting

```bash
# Format changed files
./scripts/format.sh

# Format all files
./scripts/format.sh --all

# Install lint deps
pip install -c python/requirements_compiled.txt -r python/requirements/lint-requirements.txt
```

Tools: **Black** (line length 88), **isort** (Black-compatible), **flake8**, **mypy** (selective), **clang-format** (C++), **shellcheck** (shell).

## Architecture

### C++ Core (`src/ray/`)

- **Core Worker** (`core_worker/`) — task execution, object store interaction, memory management
- **Raylet** (`raylet/`) — local scheduler, resource management, worker pool
- **GCS Server** (`gcs/`) — centralized metadata and cluster state (Global Control Store)
- **Object Manager** (`object_manager/`) — shared-memory plasma store, object transfer, spilling
- **RPC** (`rpc/`) — gRPC-based inter-process communication
- **Protobuf** (`protobuf/`) — all proto definitions for the RPC layer

### Python (`python/ray/`)

- `_raylet.pyx` — Cython bindings to C++ core (the main Python-C++ bridge)
- `_private/` — internal implementation details
- `autoscaler/` — cluster autoscaling with cloud provider backends (AWS, GCP, Azure)
- `dashboard/` — monitoring web UI backend (frontend in `dashboard/client/`)

### Ray Libraries

- **Data** (`python/ray/data/`) — distributed data processing
- **Train** (`python/ray/train/`) — distributed training
- **Tune** (`python/ray/tune/`) — hyperparameter tuning
- **Serve** (`python/ray/serve/`) — model serving
- **RLlib** (`rllib/`) — reinforcement learning (top-level directory, not under `python/ray/`)

### Other

- `java/` — Java API
- `cpp/` — C++ client library and examples
- `dashboard/client/` — React/TypeScript dashboard frontend
- `ci/` — CI scripts (Buildkite-based); `ci/run/bazel.py` orchestrates Bazel test runs
