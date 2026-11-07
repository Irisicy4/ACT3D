#!/bin/bash
# Answer judge for the RL reward: Qwen3-30B-A3B-Instruct behind an OpenAI-compatible endpoint.
#   GPUS=7 bash act3d/scripts/serve_judge.sh        # -> http://127.0.0.1:30000/v1, model "judge"
set -euo pipefail
GPUS=${GPUS:?set GPUS, e.g. GPUS=7}
PORT=${PORT:-30000}
MODEL=${JUDGE_MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
CUDA_VISIBLE_DEVICES=$GPUS VLLM_USE_DEEP_GEMM=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name judge --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$(echo "$GPUS" | awk -F, '{print NF}')" \
  --gpu-memory-utilization "${GMU:-0.60}" --max-model-len 16384 --enforce-eager
