#!/bin/bash
# Training Dashboard - Wrapper for single-run tmux sessions.
#
# One tmux session per training run (single layout, one pane per window).
# This script delegates to launch_training_tmux.sh and provides convenience
# commands for attach/kill/status.
#
# Usage:
#   ./training-dashboard.sh 0.6B          # Create a run session
#   ./training-dashboard.sh attach 0.6B   # Attach to a session
#   ./training-dashboard.sh kill 0.6B     # Kill a session
#   ./training-dashboard.sh status        # Quick status check (no tmux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launch_training_tmux.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print usage
print_usage() {
    cat << EOF
Usage: $(basename "$0") <command> [model]

Commands:
  0.6B              Create a run session for Qwen3-0.6B
  1.7B              Create a run session for Qwen3-1.7B
  4B                Create a run session for Qwen3-4B
  attach <model>    Attach to a model's tmux session
  kill <model>      Kill a model's tmux session
  status            Quick status check (no tmux)
  help              Show this help message

Examples:
  $(basename "$0") 0.6B              # Start 0.6B training
  $(basename "$0") attach 1.7B       # Attach to 1.7B session
  $(basename "$0") status            # Quick status
  $(basename "$0") kill 4B           # Kill 4B session
EOF
}

# Main
main() {
    case "${1:-}" in
        "0.6B"|"1.7B"|"4B")
            "$LAUNCHER" "$1"
            ;;
        "attach")
            if [ -z "${2:-}" ]; then
                log_error "Please specify a model: attach 0.6B|1.7B|4B"
                exit 1
            fi
            tmux attach -t "qwen-${2,,}"
            ;;
        "kill")
            if [ -z "${2:-}" ]; then
                log_error "Please specify a model: kill 0.6B|1.7B|4B"
                exit 1
            fi
            "$LAUNCHER" kill "$2"
            ;;
        "status")
            "$LAUNCHER" status
            ;;
        "-h"|"--help"|"help")
            print_usage
            ;;
        *)
            log_error "Unknown command: $1"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
