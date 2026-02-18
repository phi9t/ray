set shell := ["bash", "-euo", "pipefail", "-c"]

zephyr := "dev/zephyr/zephyr_ray.sh"

[private]
default:
    @just --list

# Verify the Zephyr wrapper and run container preflight.
check:
    @test -x "{{ zephyr }}"
    @{{ zephyr }} preflight

# Start the Zephyr container.
up:
    @{{ zephyr }} up

# Stop the Zephyr container.
down:
    @{{ zephyr }} down

# Show Zephyr container status.
status:
    @{{ zephyr }} status

# Stream Zephyr container logs.
logs:
    @{{ zephyr }} logs

# Open an interactive shell in Zephyr with the Ray venv when available.
shell:
    @{{ zephyr }} shell

# Run the Zephyr preflight stage.
preflight:
    @{{ zephyr }} preflight

# Print the active Zephyr toolchain environment.
debug-env:
    @{{ zephyr }} debug-env

# Build Ray with Bazel inside Zephyr.
build:
    @{{ zephyr }} build

# Reuse the existing Ray uv environment if present.
install:
    @{{ zephyr }} install --venv-reuse

# Recreate the Ray uv environment from scratch.
install-clear:
    @{{ zephyr }} install --venv-clear

# Run the Ray smoke workload.
smoke:
    @{{ zephyr }} run-smoke

# Run the standard end-to-end workload.
e2e:
    @{{ zephyr }} run-e2e

# Run the comprehensive Zephyr validation workflow.
e2e-comprehensive:
    @{{ zephyr }} run-e2e-comprehensive

# Run the strict release-ready validation gate.
release:
    @{{ zephyr }} release-ready
