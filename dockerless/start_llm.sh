#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

uv tool run vllm@v0.9.1 serve \
  --model=google/gemma-3-1b-it \
  --max-model-len=16384 \
  --dtype=bfloat16 \
  --gpu-memory-utilization=0.5 \
  --port=8091
