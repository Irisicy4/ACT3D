#!/bin/bash
# Stage 2: GRPO with the verifiable reward (paper Sec. 3.3). Needs the judge (serve_judge.sh).
#   MODEL=checkpoints/act3d_sft_hf GPUS=0,1,2,3 bash act3d/scripts/train_rl.sh
# Extra hydra overrides are passed through, e.g. trainer.n_gpus_per_node=8.
set -euo pipefail
: "${ACT3D_ROOT:?set ACT3D_ROOT}"
MODEL=${MODEL:?set MODEL to the merged SFT checkpoint}
GPUS=${GPUS:-0,1,2,3}
OUT=${OUT:-checkpoints/act3d_rl}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

export ACT3D_ROOT
export ACT3D_DIR=$REPO/act3d
export ACT3D_PKG=$ACT3D_DIR/act3d
export PYTHONPATH=$ACT3D_DIR:$REPO:${PYTHONPATH:-}
export VERL_NO_TOOL_SCHEMA=1   # the system prompt describes the tool; no `# Tools` block
export JUDGE_BASE=${JUDGE_BASE:-http://127.0.0.1:30000/v1} JUDGE_MODEL=${JUDGE_MODEL:-judge}
export CUDA_VISIBLE_DEVICES=$GPUS

# The reward assumes one working judge for the whole run; a judge failure mid-run stops the step.
curl -sf -m 10 "$JUDGE_BASE/models" | grep -q "\"$JUDGE_MODEL\"" \
  || { echo "judge '$JUDGE_MODEL' not reachable at $JUDGE_BASE (start it with serve_judge.sh)" >&2; exit 1; }

# Ray workers get this job's environment explicitly (a pre-started Ray head would not pass it on).
# The single quotes reach hydra, which then keeps every value a string.
RT_ENV=()
for k in ACT3D_ROOT ACT3D_DIR ACT3D_PKG PYTHONPATH VERL_NO_TOOL_SCHEMA JUDGE_BASE JUDGE_MODEL; do
  RT_ENV+=("+ray_kwargs.ray_init.runtime_env.env_vars.$k='${!k}'")
done

python -m verl.trainer.main_ppo \
  --config-path="$ACT3D_DIR/configs" --config-name=act3d_grpo \
  data.train_files="$ACT3D_ROOT/rl/train.parquet" \
  data.val_files="[$ACT3D_ROOT/rl/val.parquet]" \
  actor_rollout_ref.model.path="$MODEL" \
  trainer.n_gpus_per_node="$(echo "$GPUS" | awk -F, '{print NF}')" \
  trainer.experiment_name=rl trainer.default_local_dir="$OUT" \
  "${RT_ENV[@]}" "$@"
