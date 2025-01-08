#!/bin/bash

set -eu -o pipefail

# This needed to be created separately.
readonly CONTAINER_IMAGE="sygaldry/ray_builder:base"
readonly HOST_WORKSPACE="/mnt/data_infra/workspace/ray"
mkdir -p "${HOST_WORKSPACE}"
readonly WORKSPACE="/host/workspace"

readonly HOST_HOME_DIR="${HOST_WORKSPACE}/container_home"
mkdir -p "${HOST_HOME_DIR}"
readonly HOME_DIR="/home/${USER}"

readonly SPACK_BRANCH="v0.23.0"
readonly HOST_SPACK_BUILDER_BASE="${HOST_WORKSPACE}/spack_builder_base"
mkdir -p "${HOST_SPACK_BUILDER_BASE}"
readonly SPACK_BUILDER_BASE="/opt/spack_builder_base"

# This directory is set in "base.dockerfile"
readonly HOST_CCACHE_BASE="${HOST_WORKSPACE}/opt_ccache"
mkdir -p "${HOST_CCACHE_BASE}"
readonly CCACHE_BASE="/opt/ccache"

readonly HOST_CODE_BASE="${HOME}/CodeBase"
readonly CODE_BASE="/host/codebase"

readonly HOST_SPACK_REPO="${HOST_WORKSPACE}/spack"
readonly SPACK_REPO="${WORKSPACE}/spack"

readonly HOST_ENTRYPOINT_PATH="${HOST_WORKSPACE}/entrypoint.sh"
readonly ENTRYPOINT_PATH="${WORKSPACE}/entrypoint.sh"

if [[ ! -d "${HOST_WORKSPACE}/spack" ]]; then
    cat <<_SPACK_INSTALL_EOF_
could not find containerized spack repository
use this to create it
git clone -c feature.manyFiles=true https://github.com/spack/spack.git ${HOST_SPACK_REPO}
_SPACK_INSTALL_EOF_
    git clone -c feature.manyFiles=true https://github.com/spack/spack.git --branch "${SPACK_BRANCH}" --single-branch "${HOST_SPACK_REPO}"
fi

cat <<_SCRIPT_HEADER_EOF_ | tee "${HOST_ENTRYPOINT_PATH}"
#!/bin/bash

set -eu -o pipefail

# ======== CONSTANTS =======
readonly SPACK_REPO="${SPACK_REPO}"
readonly WORKSPACE="${WORKSPACE}"
readonly ENTRYPOINT_PATH="${ENTRYPOINT_PATH}"
# ==========================

_SCRIPT_HEADER_EOF_

cat <<'_SCRIPT_BODY_EOF_' | tee -a "${HOST_ENTRYPOINT_PATH}"

source "${SPACK_REPO}/share/spack/setup-env.sh"
source "${SPACK_REPO}/share/spack/spack-completion.bash"

exec bash

_SCRIPT_BODY_EOF_

chmod a+x "${HOST_ENTRYPOINT_PATH}"
echo "${HOST_ENTRYPOINT_PATH}"

docker run \
    --gpus all \
    --init \
    --rm \
    --interactive \
    --tty \
    --net host \
    --ipc host \
    --volume /etc/passwd:/etc/passwd:ro \
    --volume /etc/group:/etc/group:ro \
    --volume /etc/shadow:/etc/shadow:ro \
    -u "$(id -u):$(id -g)" \
    --env SPACK_CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    -v "${HOST_HOME_DIR}:${HOME_DIR}" \
    -v "${HOST_WORKSPACE}:${WORKSPACE}" \
    -v "${HOST_SPACK_BUILDER_BASE}:${SPACK_BUILDER_BASE}" \
    -v "${HOST_CCACHE_BASE}:${CCACHE_BASE}" \
    -v "${HOST_ENTRYPOINT_PATH}:${ENTRYPOINT_PATH}:ro" \
    -v "${HOST_CODE_BASE}:${CODE_BASE}" \
    -w "${CODE_BASE}/ray/spack" \
    --entrypoint="${ENTRYPOINT_PATH}" \
    "${CONTAINER_IMAGE}"
