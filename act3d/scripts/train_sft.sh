#!/bin/bash
# Stage 1: trajectory-supervised fine-tuning (paper Sec. 3.2). Loss on the model's own tokens
# (reasoning, actions, answer); system, question and observations are masked.
#   MODEL=/path/to/Qwen3.5-9B GPUS=0,1,2,3 bash act3d/scripts/train_sft.sh
set -euo pipefail
: "${ACT3D_ROOT:?set ACT3D_ROOT}"
MODEL=${MODEL:?set MODEL to the Qwen3.5-9B checkpoint}
GPUS=${GPUS:-0,1,2,3}
OUT=${OUT:-checkpoints/act3d_sft}
NG=$(echo "$GPUS" | awk -F, '{print NF}')

TRAIN=$(ls "$ACT3D_ROOT"/sft/train*.parquet | paste -sd, -)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS \
torchrun --standalone --nnodes=1 --nproc-per-node="$NG" -m verl.trainer.sft_trainer \
  data.train_files="[$TRAIN]" data.val_files="[$ACT3D_ROOT/sft/val.parquet]" \
  data.train_batch_size=16 data.micro_batch_size_per_gpu=1 data.max_length=80000 \
  data.use_dynamic_bsz=False data.truncation=error \
  data.ignore_input_ids_mismatch=False \
  model.path="$MODEL" model.enable_gradient_checkpointing=True \
  +model.override_config.attn_implementation=sdpa \
  engine=fsdp optim=fsdp optim.lr=1e-5 engine.strategy=fsdp2 engine.fsdp_size="$NG" \
  engine.param_offload=True engine.optimizer_offload=True \
  trainer.total_epochs=2 trainer.save_freq=50 trainer.test_freq=50 trainer.max_ckpt_to_keep=4 \
  trainer.logger=[console] trainer.project_name=act3d trainer.experiment_name=sft \
  trainer.default_local_dir="$OUT" "$@"
