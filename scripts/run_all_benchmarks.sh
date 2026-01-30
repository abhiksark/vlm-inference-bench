#!/bin/bash
# scripts/run_all_benchmarks.sh
# Run all VLM benchmarks sequentially (single GPU)

set -e
cd "$(dirname "$0")/.."

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    ./scripts/cleanup_gpu.sh
}
trap cleanup EXIT

# Wait for endpoint
wait_ready() {
    local url=$1
    local max_wait=$2
    for i in $(seq 1 $max_wait); do
        curl -s "$url" > /dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

echo "=== VLM Benchmark Suite ==="
echo "Date: $(date)"
echo ""

# Initial cleanup
./scripts/cleanup_gpu.sh

# 1. Ollama
echo -e "\n[1/3] Ollama"
docker build -t docker-ollama -f docker/ollama/Dockerfile docker/ollama/
docker run -d --gpus all --name vlm-bench-ollama -p 11434:11434 -v ollama_data:/root/.ollama docker-ollama
echo "Waiting for Ollama service..."
wait_ready "http://localhost:11434/api/tags" 60
echo "Pulling models (if needed)..."
docker exec vlm-bench-ollama ollama pull qwen3-vl:4b
docker exec vlm-bench-ollama ollama pull qwen3-vl:4b-thinking
echo "Running benchmarks..."
python src/benchmark.py --config configs/benchmark_ollama.yaml
python src/benchmark.py --config configs/benchmark_ollama_thinking.yaml
docker stop vlm-bench-ollama && docker rm vlm-bench-ollama

# 2. SGLang
echo -e "\n[2/3] SGLang"
docker build -t docker-sglang -f docker/sglang/Dockerfile docker/sglang/
docker run -d --gpus all --name vlm-bench-sglang -p 8004:8004 -v ~/.cache/huggingface:/root/.cache/huggingface docker-sglang
echo "Waiting for SGLang to be ready..."
wait_ready "http://localhost:8004/v1/models" 120
python src/benchmark.py --config configs/benchmark_sglang.yaml
python src/benchmark.py --config configs/benchmark_sglang_thinking.yaml
docker stop vlm-bench-sglang && docker rm vlm-bench-sglang

# 3. vLLM
echo -e "\n[3/3] vLLM"
docker build -t docker-vllm-awq -f docker/vllm-awq/Dockerfile docker/vllm-awq/
docker run -d --gpus all --name vlm-bench-vllm-awq -p 8001:8001 -v ~/.cache/huggingface:/root/.cache/huggingface docker-vllm-awq
echo "Waiting for vLLM to be ready..."
wait_ready "http://localhost:8001/v1/models" 120

# Run benchmarks: normal, thinking, then warm runs (KV cache populated)
echo "  - Normal mode (cold)"
python src/benchmark.py --config configs/benchmark_vllm.yaml
echo "  - Thinking mode"
python src/benchmark.py --config configs/benchmark_vllm_thinking.yaml
echo "  - Normal mode (warm - KV cache populated)"
python src/benchmark.py --config configs/benchmark_vllm.yaml
echo "  - Thinking mode (warm - KV cache populated)"
python src/benchmark.py --config configs/benchmark_vllm_thinking.yaml

docker stop vlm-bench-vllm-awq && docker rm vlm-bench-vllm-awq

echo -e "\n=== All benchmarks complete! ==="
echo "Results saved to: results/"
