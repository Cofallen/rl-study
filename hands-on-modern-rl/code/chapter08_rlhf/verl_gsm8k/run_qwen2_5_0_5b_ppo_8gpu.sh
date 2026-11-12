#!/bin/bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
export ROLLOUT_N=${ROLLOUT_N:-4}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TEST_FREQ=${TEST_FREQ:-2}

"$SCRIPT_DIR/run_qwen2_5_0_5b_ppo_single_gpu.sh" "$@"

