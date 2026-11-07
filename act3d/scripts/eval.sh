#!/bin/bash
# Six-benchmark interactive evaluation of a Hugging Face checkpoint: one vLLM server per GPU,
# each benchmark sharded across the servers, then scored.
#   MODEL=checkpoints/act3d_rl_hf GPUS=0,1,2,3 OUT=out/act3d bash act3d/scripts/eval.sh
set -uo pipefail
: "${ACT3D_ROOT:?set ACT3D_ROOT}"
MODEL=${MODEL:?set MODEL}; GPUS=${GPUS:-0}; OUT=${OUT:-out/eval}
BENCHES="scanqa sqa3d spbench_mv spbench_si mp3d arkit"
mkdir -p "$OUT"
[ -d "$MODEL" ] && MODEL=$(cd "$MODEL" && pwd)   # local checkpoint dir (an HF id passes through)
OUT=$(cd "$OUT" && pwd)
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
IFS=, read -ra G <<< "$GPUS"; N=${#G[@]}

pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 \
    --port $((31000 + i)) --model "$MODEL" --served-model-name act3d --gpu-memory-utilization 0.85 \
    --max-model-len 65536 --enforce-eager --limit-mm-per-prompt.image 20 > "$OUT/server_$i.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
for i in "${!G[@]}"; do   # wait for each server; give up if it exits or is not up in 30 min
  for t in $(seq 180); do
    curl -s -m 3 "127.0.0.1:$((31000 + i))/v1/models" | grep -q act3d && continue 2
    kill -0 "${pids[$i]}" 2>/dev/null || break
    sleep 10
  done
  echo "vLLM server on GPU ${G[$i]} failed to start; see $OUT/server_$i.log" >&2; exit 1
done

for b in $BENCHES; do
  run=()
  for i in $(seq 0 $((N - 1))); do   # shard i = every N-th question
    awk -v n="$N" -v i="$i" 'NR % n == i' "$ACT3D_ROOT/eval/$b.jsonl" > "$OUT/$b.pool$i.jsonl"
    python eval/run_eval.py --pool "$OUT/$b.pool$i.jsonl" --out "$OUT/$b.part$i.jsonl" \
      --base "http://127.0.0.1:$((31000 + i))/v1" --model act3d > "$OUT/$b.part$i.log" 2>&1 &
    run+=($!)
  done
  for p in "${run[@]}"; do
    wait "$p" || { echo "$b: an evaluation shard failed; see $OUT/$b.part*.log" >&2; exit 1; }
  done
  cat "$OUT/$b".part*.jsonl > "$OUT/$b.jsonl"
done
python eval/score.py --pools "$ACT3D_ROOT/eval" --run "$OUT"
