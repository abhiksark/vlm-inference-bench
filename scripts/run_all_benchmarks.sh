#!/bin/bash
# scripts/run_all_benchmarks.sh
# Run all VLM benchmarks organized by precision group (single GPU)

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

# Create results directories
mkdir -p results/bf16 results/fp8 results/8bit results/4bit

# Initial cleanup
./scripts/cleanup_gpu.sh

# ============================================================
# GROUP 1: BFloat16 (Full Precision)
# ============================================================
echo -e "\n=========================================="
echo "GROUP 1: BFloat16 (Full Precision)"
echo "=========================================="

# 1a. vLLM BF16
echo -e "\n[1a/10] vLLM BF16"
docker build -t docker-vllm-bf16 -f docker/vllm-awq/Dockerfile docker/vllm-awq/
docker run -d --gpus all --name vlm-bench-vllm-bf16 -p 28001:28001 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-vllm-bf16
wait_ready "http://localhost:28001/v1/models" 120
python src/benchmark.py --config configs/bf16/benchmark_vllm_bf16.yaml
docker stop vlm-bench-vllm-bf16 && docker rm vlm-bench-vllm-bf16

# 1b. SGLang BF16
echo -e "\n[1b/10] SGLang BF16"
docker build -t docker-sglang-bf16 -f docker/sglang/Dockerfile docker/sglang/
docker run -d --gpus all --name vlm-bench-sglang-bf16 -p 28004:28004 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-sglang-bf16
wait_ready "http://localhost:28004/v1/models" 120
python src/benchmark.py --config configs/bf16/benchmark_sglang_bf16.yaml
docker stop vlm-bench-sglang-bf16 && docker rm vlm-bench-sglang-bf16

# 1c. Ollama F16
echo -e "\n[1c/10] Ollama F16"
docker build -t docker-ollama -f docker/ollama/Dockerfile docker/ollama/
docker run -d --gpus all --name vlm-bench-ollama -p 28434:28434 \
    -v ollama_data:/root/.ollama docker-ollama
wait_ready "http://localhost:28434/api/tags" 60
echo "Pulling F16 model..."
docker exec vlm-bench-ollama ollama pull qwen3-vl:4b-fp16
python src/benchmark.py --config configs/bf16/benchmark_ollama_bf16.yaml
docker stop vlm-bench-ollama && docker rm vlm-bench-ollama

# ============================================================
# GROUP 2: FP8 Quantization
# ============================================================
echo -e "\n=========================================="
echo "GROUP 2: FP8 Quantization (W8A16 on Ampere)"
echo "=========================================="

# 2a. vLLM FP8
echo -e "\n[2a/10] vLLM FP8"
docker build -t docker-vllm-fp8 -f docker/vllm-fp8/Dockerfile docker/vllm-fp8/
docker run -d --gpus all --name vlm-bench-vllm-fp8 -p 28002:28002 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-vllm-fp8
wait_ready "http://localhost:28002/v1/models" 120
python src/benchmark.py --config configs/fp8/benchmark_vllm_fp8.yaml
docker stop vlm-bench-vllm-fp8 && docker rm vlm-bench-vllm-fp8

# 2b. SGLang FP8
echo -e "\n[2b/10] SGLang FP8"
docker build -t docker-sglang-fp8 -f docker/sglang-fp8/Dockerfile docker/sglang-fp8/
docker run -d --gpus all --name vlm-bench-sglang-fp8 -p 28005:28005 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-sglang-fp8
wait_ready "http://localhost:28005/v1/models" 120
python src/benchmark.py --config configs/fp8/benchmark_sglang_fp8.yaml
docker stop vlm-bench-sglang-fp8 && docker rm vlm-bench-sglang-fp8

# ============================================================
# GROUP 3: 8-bit Quantization
# ============================================================
echo -e "\n=========================================="
echo "GROUP 3: 8-bit Quantization"
echo "=========================================="

# 3a. vLLM AWQ 8-bit
echo -e "\n[3a/10] vLLM AWQ 8-bit"
docker build -t docker-vllm-awq8 -f docker/vllm-awq8/Dockerfile docker/vllm-awq8/
docker run -d --gpus all --name vlm-bench-vllm-awq8 -p 28003:28003 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-vllm-awq8
wait_ready "http://localhost:28003/v1/models" 120
python src/benchmark.py --config configs/8bit/benchmark_vllm_awq8.yaml
docker stop vlm-bench-vllm-awq8 && docker rm vlm-bench-vllm-awq8

# 3b. SGLang AWQ 8-bit
echo -e "\n[3b/10] SGLang AWQ 8-bit"
docker build -t docker-sglang-awq8 -f docker/sglang-awq8/Dockerfile docker/sglang-awq8/
docker run -d --gpus all --name vlm-bench-sglang-awq8 -p 28006:28006 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-sglang-awq8
wait_ready "http://localhost:28006/v1/models" 120
python src/benchmark.py --config configs/8bit/benchmark_sglang_awq8.yaml
docker stop vlm-bench-sglang-awq8 && docker rm vlm-bench-sglang-awq8

# 3c. Ollama Q8_0
echo -e "\n[3c/10] Ollama Q8_0"
docker build -t docker-ollama -f docker/ollama/Dockerfile docker/ollama/
docker run -d --gpus all --name vlm-bench-ollama -p 28434:28434 \
    -v ollama_data:/root/.ollama docker-ollama
wait_ready "http://localhost:28434/api/tags" 60
echo "Pulling Q8_0 model..."
docker exec vlm-bench-ollama ollama pull qwen3-vl:4b-q8_0
python src/benchmark.py --config configs/8bit/benchmark_ollama_q8.yaml
docker stop vlm-bench-ollama && docker rm vlm-bench-ollama

# ============================================================
# GROUP 4: 4-bit Quantization (Ollama only)
# ============================================================
echo -e "\n=========================================="
echo "GROUP 4: 4-bit Quantization (Ollama only)"
echo "=========================================="

# 4a. Ollama Q4_K_M
echo -e "\n[4a/10] Ollama Q4_K_M"
docker build -t docker-ollama -f docker/ollama/Dockerfile docker/ollama/
docker run -d --gpus all --name vlm-bench-ollama -p 28434:28434 \
    -v ollama_data:/root/.ollama docker-ollama
wait_ready "http://localhost:28434/api/tags" 60
echo "Pulling Q4_K_M model..."
docker exec vlm-bench-ollama ollama pull qwen3-vl:4b
python src/benchmark.py --config configs/4bit/benchmark_ollama_q4.yaml
docker stop vlm-bench-ollama && docker rm vlm-bench-ollama

echo -e "\n=========================================="
echo "=== All benchmarks complete! ==="
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/bf16/  (BFloat16 comparisons)"
echo "  - results/fp8/   (FP8 comparisons)"
echo "  - results/8bit/  (8-bit comparisons)"
echo "  - results/4bit/  (4-bit Ollama)"
