#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="${SCRIPT_DIR}/zephyr_ray.sh"

[[ -x "${ORCH}" ]] || {
  echo "ERROR: missing executable orchestrator: ${ORCH}" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Compatibility wrapper for legacy Zephyr entry command.

Preferred:
  dev/zephyr/zephyr_ray.sh shell|status|logs|down

Supported legacy options:
  dev/zephyr/enter_zephyr_ray.sh
  dev/zephyr/enter_zephyr_ray.sh --status
  dev/zephyr/enter_zephyr_ray.sh --logs
  dev/zephyr/enter_zephyr_ray.sh --down
USAGE
}

case "${1:-}" in
  "")
    exec "${ORCH}" shell
    ;;
  --status)
    exec "${ORCH}" status
    ;;
  --logs)
    exec "${ORCH}" logs
    ;;
  --down)
    exec "${ORCH}" down
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "ERROR: unknown option: ${1}" >&2
    usage
    exit 1
    ;;
esac
