FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/data/huggingface

# System dependencies (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    cmake gcc g++ git curl wget pkg-config \
    libjemalloc-dev libunwind-dev \
    psmisc unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Bazelisk (provides Bazel 6.5.0 via .bazelversion)
RUN curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create venv with uv
RUN uv venv /opt/venv --python python3.10
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python build deps (pip needed by Ray's build backend)
RUN uv pip install pip cython wheel setuptools numpy

# PyTorch (cu121)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# ML dependencies
RUN uv pip install \
    transformers datasets accelerate peft \
    tensorboard scikit-learn pandas pyarrow

WORKDIR /workspace/ray
