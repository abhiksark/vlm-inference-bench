#!/bin/bash
# scripts/cleanup_gpu.sh
# Stop VLM containers and free GPU RAM

echo "=== GPU Memory Before ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo -e "\n=== Stopping VLM containers ==="
for name in vlm-bench-vllm-bf16 vlm-bench-vllm-fp8 vlm-bench-vllm-awq8 \
            vlm-bench-sglang-bf16 vlm-bench-sglang-fp8 vlm-bench-sglang-awq8 \
            vlm-bench-ollama vlm-bench-vllm-awq vlm-bench-sglang qwen3-vl-server; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "Stopping $name..."
        docker stop "$name" 2>/dev/null && docker rm "$name" 2>/dev/null
    fi
done

echo -e "\n=== GPU Memory After ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
