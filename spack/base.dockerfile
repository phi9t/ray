ARG RAY_BASE_IMAGE="rayproject/ray:2.24.0-py311-cu121"
FROM ${RAY_BASE_IMAGE}

# https://spack.readthedocs.io/en/latest/getting_started.html
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        ccache \
        cmake \
        coreutils \
        curl \
        environment-modules \
        g++ \
        gfortran \
        git \
        gpg \
        gzip \
        lsb-release \
        pkg-config \
        python3 \
        python3-distutils \
        python3-venv \
        unzip \
        wget \
        zip \
    && \
    rm -rf /var/lib/apt/lists/*

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

USER ${RAY_UID}
RUN conda install -y jupyterlab
RUN conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda install -y conda-forge::transformers

ARG BAZELISK_VERSION=1.20.0
ENV BAZELISK_VERSION="${BAZELISK_VERSION}"
RUN wget "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64" -O /usr/local/bin/bazel \
    && chmod a+x /usr/local/bin/bazel
