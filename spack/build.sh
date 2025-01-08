#!/bin/bash

set -eu -o pipefail

if [[ -z "${SPACK_CONTAINER_IMAGE:-}" ]]; then
    echo "This script should be run inside a Spack container"
    exit 1
fi

cp spack_src.yaml spack.yaml

readonly SPACK_BUILDER_BASE="/opt/spack_builder_base/pkg/ray/spack"
cat <<_YAML_SUFFIX_EOF_ | tee -a spack.yaml

  # <spack_git_repo>/etc/spack/defaults
  config:
    install_tree:
       root: ${SPACK_BUILDER_BASE}/install_tree
    build_stage:
      - ${SPACK_BUILDER_BASE}/build_stage
    template_dirs:
      - ${SPACK_BUILDER_BASE}/template_dirs
    license_dir: ${SPACK_BUILDER_BASE}/license_dir
    test_stage: ${SPACK_BUILDER_BASE}/test_stage
    source_cache: ${SPACK_BUILDER_BASE}/source_cache
    misc_cache: ${SPACK_BUILDER_BASE}/misc_cache

_YAML_SUFFIX_EOF_


spack --env . concretize --force
spack --env . install
