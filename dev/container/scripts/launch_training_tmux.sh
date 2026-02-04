#!/bin/bash
# Launch long-term Qwen3 LoRA training in tmux sessions.
#
# Creates a tmux session per model with one pane per window:
#   - train:  Training job (docker exec)
#   - gpu:    nvitop (in-container, focused on run)
#   - system: htop (in-container, focused on run)
#   - metrics: tail metrics.jsonl with jq formatting
#   - status: live run summary (no tmux required to inspect)
#   - shell:  container shell for manual commands
#
# Usage:
#   ./launch_training_tmux.sh 0.6B    # Launch 0.6B training
#   ./launch_training_tmux.sh 1.7B    # Launch 1.7B training
#   ./launch_training_tmux.sh 4B      # Launch 4B training
#   ./launch_training_tmux.sh all     # Launch all (sequentially)
#   ./launch_training_tmux.sh status  # Check status of all sessions
#   ./launch_training_tmux.sh attach 0.6B  # Attach to a session
#
# Environment:
#   DOCKER_COMPOSE_FILE: Path to docker-compose.yml (auto-detected)
#   TRAINING_NUM_STEPS: Override default 10000 steps
#   TRAINING_EVAL_STEPS: Override default 100 eval steps
#   TRAINING_DATA_DIR: Data directory (for run manifest)
#   TRAINING_CONFIG_PATH: Config path (for run manifest)
#   TRAINING_CKPT_DIR: Checkpoint directory (for run manifest)
#   TRAINING_LOG_DIR: Log directory (for run manifest)
#   TRAINING_MONITOR_IN_CONTAINER: 1 to run nvitop/htop inside container (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-detect docker-compose file
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$CONTAINER_DIR/docker-compose.yml}"

# Training parameters (can be overridden via env vars)
NUM_STEPS="${TRAINING_NUM_STEPS:-10000}"
EVAL_STEPS="${TRAINING_EVAL_STEPS:-100}"
SAVE_STEPS="${TRAINING_SAVE_STEPS:-500}"

# Output directories (host path for metrics tailing)
HOST_OUTPUT_DIR="/mnt/data_infra/shared/outputs"
REPO_ROOT="$(cd "$CONTAINER_DIR/../.." && pwd)"

# Run manifest defaults (for tracking assets/dirs)
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-}"
TRAINING_CONFIG_PATH="${TRAINING_CONFIG_PATH:-}"
TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-}"
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-}"
TRAINING_MONITOR_IN_CONTAINER="${TRAINING_MONITOR_IN_CONTAINER:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check prerequisites
check_prereqs() {
    if ! command -v tmux &> /dev/null; then
        log_error "tmux is not installed. Please install: sudo apt install tmux"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    if ! command -v jq &> /dev/null; then
        log_warn "jq is not installed. Metrics window will be less readable."
    fi

    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found at: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
}

# Get session name for a model
get_session_name() {
    local model="$1"
    echo "qwen-${model,,}"  # lowercase
}

# Get training command for a model
get_training_cmd() {
    local model="$1"
    local output_dir="/data/outputs/qwen3-longrun-${model,,}"

    case "$model" in
        "0.6B")
            # DDP mode: use torchrun with 2 GPUs
            echo "torchrun --nproc_per_node=2 /workspace/ray/dev/container/scripts/run_long_training.py --model 0.6B --num_steps $NUM_STEPS --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS --output_dir $output_dir"
            ;;
        "1.7B"|"4B")
            # Single process (handled by the script)
            echo "python /workspace/ray/dev/container/scripts/run_long_training.py --model $model --num_steps $NUM_STEPS --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS --output_dir $output_dir"
            ;;
        *)
            log_error "Unknown model: $model"
            exit 1
            ;;
    esac
}

# Write a run manifest capturing assets/dirs/data for the training run.
write_run_manifest() {
    local model="$1"
    local session_name="$2"
    local output_dir_host="$3"
    local output_dir_container="$4"
    local metrics_file="$5"
    local pid_file_host="$6"
    local pgid_file_host="$7"
    local train_cmd="$8"
    local start_time
    local host
    local user
    local git_sha
    local pid_file_container="$output_dir_container/training.pid"
    local pgid_file_container="$output_dir_container/training.pgid"

    start_time="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    host="$(hostname)"
    user="${USER:-unknown}"
    git_sha="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

    mkdir -p "$output_dir_host"

    cat > "$output_dir_host/run.info" << EOF
run_id=$session_name
model=$model
start_time_utc=$start_time
host=$host
user=$user
git_sha=$git_sha
docker_compose_file=$DOCKER_COMPOSE_FILE
container_service=ray-dev
output_dir_host=$output_dir_host
output_dir_container=$output_dir_container
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}
data_dir=${TRAINING_DATA_DIR:-}
config_path=${TRAINING_CONFIG_PATH:-}
ckpt_dir=${TRAINING_CKPT_DIR:-}
log_dir=${TRAINING_LOG_DIR:-}
metrics_file=$metrics_file
pid_file_host=$pid_file_host
pgid_file_host=$pgid_file_host
pid_file_container=$pid_file_container
pgid_file_container=$pgid_file_container
train_cmd=$train_cmd
EOF
}

# Build the command that runs inside the container and records pid/pgid.
get_train_wrapper_cmd() {
    local train_cmd="$1"
    local output_dir_container="$2"
    local pid_file="$3"
    local pgid_file="$4"

    cat << EOF
set -e
mkdir -p "$output_dir_container"
echo "Starting training..."
($train_cmd) & pid=\$!
echo "\$pid" > "$pid_file"
pgid=\$(ps -o pgid= -p "\$pid" | tr -d ' ')
echo "\$pgid" > "$pgid_file"
echo "PID=\$pid PGID=\$pgid"
wait "\$pid"
EOF
}

# Build a command to run htop focused on the training process tree.
get_htop_cmd() {
    local pgid_file="$1"
    cat << 'EOF'
while [ ! -f "PGID_FILE" ]; do sleep 1; done
pgid=$(cat "PGID_FILE" 2>/dev/null)
if [ -n "$pgid" ]; then
    pids=$(pgrep -g "$pgid" | paste -sd, -)
fi
if command -v htop &> /dev/null; then
    if [ -n "$pids" ]; then
        htop -p "$pids"
    else
        htop
    fi
else
    top
fi
EOF
}

# Build a status loop for a single run (no tmux required).
get_status_cmd() {
    local model="$1"
    local output_dir="$2"
    local metrics_file="$3"
    local show_run_info="${4:-1}"

    cat << EOF
while true; do
    clear
    echo "Training Status (Qwen3-$model)"
    echo "==============================="
    echo "Updated: \$(date '+%H:%M:%S')"
    echo ""
    if [ "$show_run_info" = "1" ] && [ -f "$output_dir/run.info" ]; then
        echo "[run.info]"
        sed -n '1,12p' "$output_dir/run.info"
        echo ""
    fi
    if [ -f "$metrics_file" ]; then
        last=\$(tail -1 "$metrics_file" 2>/dev/null)
        if [ -n "\$last" ]; then
            step=\$(echo "\$last" | jq -r '.step // "?"' 2>/dev/null || echo "?")
            loss=\$(echo "\$last" | jq -r '.train_loss // .eval_loss // "?"' 2>/dev/null || echo "?")
            tok_s=\$(echo "\$last" | jq -r '.tokens_per_sec // "-"' 2>/dev/null || echo "-")
            echo "Latest metrics: step=\$step loss=\${loss:0:8} tok/s=\${tok_s:0:6}"
        fi
        mtime=\$(stat -c %Y "$metrics_file" 2>/dev/null || echo 0)
        now=\$(date +%s)
        if [ "\$mtime" -gt 0 ]; then
            age=\$((now - mtime))
            echo "Last update: \${age}s ago"
            if [ "\$age" -gt 120 ]; then
                echo "WARNING: metrics stale (>120s)"
            fi
        fi
    else
        echo "Metrics: waiting for $metrics_file"
    fi
    echo ""
    sleep 10
done
EOF
}

# Launch training for a specific model
launch_model() {
    local model="$1"
    local session_name=$(get_session_name "$model")
    local output_dir_host="$HOST_OUTPUT_DIR/qwen3-longrun-${model,,}"
    local metrics_file="$output_dir_host/metrics.jsonl"
    local output_dir_container="/data/outputs/qwen3-longrun-${model,,}"
    local pid_file_container="$output_dir_container/training.pid"
    local pgid_file_container="$output_dir_container/training.pgid"
    local pid_file_host="$output_dir_host/training.pid"
    local pgid_file_host="$output_dir_host/training.pgid"

    log_info "Launching training for Qwen3-$model"
    log_info "Session name: $session_name"
    log_info "Output directory: $output_dir_host"

    # Check if session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_warn "Session '$session_name' already exists"
        read -p "Kill existing session and restart? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            tmux kill-session -t "$session_name"
            log_info "Killed existing session"
        else
            log_info "Keeping existing session. Use: tmux attach -t $session_name"
            return 0
        fi
    fi

    # Create output directory on host
    mkdir -p "$output_dir_host"

    # Get docker compose command
    local docker_cmd="docker compose -f $DOCKER_COMPOSE_FILE exec -it ray-dev"

    # Get training command
    local train_cmd=$(get_training_cmd "$model")
    local train_wrapper_cmd
    local train_wrapper_cmd_escaped
    train_wrapper_cmd="$(get_train_wrapper_cmd "$train_cmd" "$output_dir_container" "$pid_file_container" "$pgid_file_container")"
    train_wrapper_cmd_escaped="$(printf '%q' "$train_wrapper_cmd")"

    # Create tmux session with train window
    log_info "Creating tmux session: $session_name"
    tmux new-session -d -s "$session_name" -n 'train'

    # Write run manifest on host
    write_run_manifest "$model" "$session_name" "$output_dir_host" "$output_dir_container" "$metrics_file" "$pid_file_host" "$pgid_file_host" "$train_cmd"

    # Send training command to train window
    tmux send-keys -t "$session_name:train" "$docker_cmd bash -lc $train_wrapper_cmd_escaped" Enter

    # Create GPU monitoring window
    log_info "Creating GPU monitor window"
    tmux new-window -t "$session_name" -n 'gpu'
    if [ "$TRAINING_MONITOR_IN_CONTAINER" = "1" ]; then
        tmux send-keys -t "$session_name:gpu" "$docker_cmd nvitop" Enter
    else
        tmux send-keys -t "$session_name:gpu" 'nvitop' Enter
    fi

    # Create system monitoring window
    log_info "Creating system monitor window"
    tmux new-window -t "$session_name" -n 'system'
    local htop_cmd
    local htop_cmd_escaped
    if [ "$TRAINING_MONITOR_IN_CONTAINER" = "1" ]; then
        htop_cmd="$(get_htop_cmd "$pgid_file_container")"
        htop_cmd="${htop_cmd//PGID_FILE/$pgid_file_container}"
    else
        htop_cmd="$(get_htop_cmd "$pgid_file_host")"
        htop_cmd="${htop_cmd//PGID_FILE/$pgid_file_host}"
    fi
    htop_cmd_escaped="$(printf '%q' "$htop_cmd")"
    if [ "$TRAINING_MONITOR_IN_CONTAINER" = "1" ]; then
        tmux send-keys -t "$session_name:system" "$docker_cmd bash -lc $htop_cmd_escaped" Enter
    else
        tmux send-keys -t "$session_name:system" "bash -lc $htop_cmd_escaped" Enter
    fi

    # Create metrics tailing window
    log_info "Creating metrics window"
    tmux new-window -t "$session_name" -n 'metrics'

    # Wait for metrics file to exist, then tail it with jq formatting
    local metrics_file_container="$output_dir_container/metrics.jsonl"
    local metrics_cmd="echo 'Waiting for metrics file...'; while [ ! -f 'METRICS_FILE' ]; do sleep 2; done; echo 'Tailing metrics:'; if command -v jq &> /dev/null; then tail -f 'METRICS_FILE' | jq -r 'if .eval_loss then \"[Step \\(.step)] eval_loss=\\(.eval_loss)\" elif .train_loss then \"[Step \\(.step)] loss=\\(.train_loss) tok/s=\\(.tokens_per_sec // \"?\")\" else . end'; else tail -f 'METRICS_FILE'; fi"
    if [ "$TRAINING_MONITOR_IN_CONTAINER" = "1" ]; then
        local metrics_cmd_container="${metrics_cmd//METRICS_FILE/$metrics_file_container}"
        local metrics_cmd_container_escaped
        metrics_cmd_container_escaped="$(printf '%q' "$metrics_cmd_container")"
        tmux send-keys -t "$session_name:metrics" "$docker_cmd bash -lc $metrics_cmd_container_escaped" Enter
    else
        local metrics_cmd_host="${metrics_cmd//METRICS_FILE/$metrics_file}"
        tmux send-keys -t "$session_name:metrics" "$metrics_cmd_host" Enter
    fi

    # Create status window
    log_info "Creating status window"
    tmux new-window -t "$session_name" -n 'status'
    local status_cmd
    local status_cmd_escaped
    if [ "$TRAINING_MONITOR_IN_CONTAINER" = "1" ]; then
        status_cmd="$(get_status_cmd "$model" "$output_dir_container" "$metrics_file_container" "0")"
        status_cmd_escaped="$(printf '%q' "$status_cmd")"
        tmux send-keys -t "$session_name:status" "$docker_cmd bash -lc $status_cmd_escaped" Enter
    else
        status_cmd="$(get_status_cmd "$model" "$output_dir_host" "$metrics_file" "1")"
        status_cmd_escaped="$(printf '%q' "$status_cmd")"
        tmux send-keys -t "$session_name:status" "bash -lc $status_cmd_escaped" Enter
    fi

    # Create shell window
    log_info "Creating shell window"
    tmux new-window -t "$session_name" -n 'shell'
    tmux send-keys -t "$session_name:shell" "$docker_cmd bash" Enter

    # Select train window by default
    tmux select-window -t "$session_name:train"

    log_success "Training session '$session_name' created successfully"
    echo ""
    echo "  Attach to session:  tmux attach -t $session_name"
    echo "  Switch windows:     Ctrl-b + 0-5 (train/gpu/system/metrics/status/shell)"
    echo "  Detach:             Ctrl-b + d"
    echo ""
}

# Show status of all training sessions
show_status() {
    echo ""
    echo "Training Session Status"
    echo "======================="
    echo ""

    for model in "0.6B" "1.7B" "4B"; do
        local session_name=$(get_session_name "$model")
        local output_dir="$HOST_OUTPUT_DIR/qwen3-longrun-${model,,}"
        local metrics_file="$output_dir/metrics.jsonl"
        local metadata_file="$output_dir/training_metadata.json"

        echo -n "Qwen3-$model ($session_name): "

        if tmux has-session -t "$session_name" 2>/dev/null; then
            echo -e "${GREEN}RUNNING${NC}"

            # Show latest metrics if available
            if [ -f "$metrics_file" ]; then
                local last_line=$(tail -1 "$metrics_file" 2>/dev/null)
                if [ -n "$last_line" ]; then
                    local step=$(echo "$last_line" | jq -r '.step // "?"')
                    local loss=$(echo "$last_line" | jq -r '.train_loss // .eval_loss // "?"')
                    local tok_s=$(echo "$last_line" | jq -r '.tokens_per_sec // "?"')
                    echo "    Latest: step=$step, loss=$loss, tok/s=$tok_s"
                fi
            fi
        else
            if [ -f "$metadata_file" ]; then
                local status=$(jq -r '.status // "unknown"' "$metadata_file" 2>/dev/null)
                case "$status" in
                    "completed")
                        echo -e "${GREEN}COMPLETED${NC}"
                        ;;
                    "interrupted")
                        echo -e "${YELLOW}INTERRUPTED${NC}"
                        ;;
                    "failed")
                        echo -e "${RED}FAILED${NC}"
                        ;;
                    *)
                        echo -e "${YELLOW}NOT RUNNING${NC} (status: $status)"
                        ;;
                esac
            else
                echo -e "${YELLOW}NOT STARTED${NC}"
            fi
        fi
    done
    echo ""
}

# Attach to a session
attach_session() {
    local model="$1"
    local session_name=$(get_session_name "$model")

    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        log_error "Session '$session_name' does not exist"
        exit 1
    fi

    tmux attach -t "$session_name"
}

# Kill a session
kill_session() {
    local model="$1"
    local session_name=$(get_session_name "$model")

    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
        log_success "Killed session: $session_name"
    else
        log_warn "Session '$session_name' does not exist"
    fi
}

# Print usage
print_usage() {
    cat << EOF
Usage: $(basename "$0") <command> [model]

Commands:
  0.6B              Launch training for Qwen3-0.6B
  1.7B              Launch training for Qwen3-1.7B
  4B                Launch training for Qwen3-4B
  all               Launch all models (sequentially, prompts between)
  status            Show status of all training sessions
  attach <model>    Attach to a model's tmux session
  kill <model>      Kill a model's tmux session
  kill-all          Kill all training sessions

Environment Variables:
  TRAINING_NUM_STEPS   Total training steps (default: 10000)
  TRAINING_EVAL_STEPS  Eval every N steps (default: 100)
  TRAINING_SAVE_STEPS  Save every N steps (default: 500)
  TRAINING_DATA_DIR    Data directory (recorded in run manifest)
  TRAINING_CONFIG_PATH Config path (recorded in run manifest)
  TRAINING_CKPT_DIR    Checkpoint directory (recorded in run manifest)
  TRAINING_LOG_DIR     Log directory (recorded in run manifest)
  TRAINING_MONITOR_IN_CONTAINER 1 to run nvitop/htop inside container (default)

Examples:
  $(basename "$0") 0.6B              # Start 0.6B training
  $(basename "$0") status            # Check all sessions
  $(basename "$0") attach 1.7B       # Attach to 1.7B session
  $(basename "$0") kill 4B           # Kill 4B session

  TRAINING_NUM_STEPS=1000 $(basename "$0") 0.6B  # Short run
EOF
}

# Main
main() {
    check_prereqs

    case "${1:-}" in
        "0.6B"|"1.7B"|"4B")
            launch_model "$1"
            ;;
        "all")
            log_info "Launching all models sequentially"
            echo ""
            echo "This will launch training for 0.6B, 1.7B, and 4B models."
            echo "Each model uses different GPU strategies:"
            echo "  - 0.6B: DDP (both GPUs, ~1 hour for 10k steps)"
            echo "  - 1.7B: Single GPU cuda:0 (~7 hours for 10k steps)"
            echo "  - 4B: Pipeline parallel (~15 hours for 10k steps)"
            echo ""
            echo "Total estimated time: ~23 hours"
            echo ""

            for model in "0.6B" "1.7B" "4B"; do
                echo ""
                read -p "Launch Qwen3-$model training? [Y/n] " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    launch_model "$model"
                else
                    log_info "Skipping $model"
                fi
            done

            echo ""
            show_status
            ;;
        "status")
            show_status
            ;;
        "attach")
            if [ -z "${2:-}" ]; then
                log_error "Please specify a model: attach 0.6B|1.7B|4B"
                exit 1
            fi
            attach_session "$2"
            ;;
        "kill")
            if [ -z "${2:-}" ]; then
                log_error "Please specify a model: kill 0.6B|1.7B|4B"
                exit 1
            fi
            kill_session "$2"
            ;;
        "kill-all")
            for model in "0.6B" "1.7B" "4B"; do
                kill_session "$model"
            done
            ;;
        "-h"|"--help"|"help"|"")
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
