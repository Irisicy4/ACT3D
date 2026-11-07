#!/bin/bash
# FSDP checkpoint -> Hugging Face model directory.
#   bash act3d/scripts/merge.sh checkpoints/act3d_sft/global_step_600        checkpoints/act3d_sft_hf
#   bash act3d/scripts/merge.sh checkpoints/act3d_rl/global_step_100/actor   checkpoints/act3d_rl_hf
set -euo pipefail
python -m verl.model_merger merge --backend fsdp --local_dir "${1:?fsdp checkpoint dir}" --target_dir "${2:?output dir}"
