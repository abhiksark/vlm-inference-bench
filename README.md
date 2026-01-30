# VLM Benchmark Suite

Benchmarking framework for Vision-Language Models (VLMs) with multiple inference backends.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Backends](#backends)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## Overview

Benchmark video understanding capabilities across inference backends:

| Backend | Features | Docs |
|---------|----------|------|
| [vLLM](https://docs.vllm.ai/) | Prefix caching, chunked prefill | [Dockerfile](docker/vllm-awq/Dockerfile) |
| [SGLang](https://github.com/sgl-project/sglang) | FlashInfer attention, RadixAttention | [Dockerfile](docker/sglang/Dockerfile) |
| [Ollama](https://ollama.ai/) | GGUF quantization (Q4_K_M) | [Dockerfile](docker/ollama/Dockerfile) |

Model: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (normal + thinking modes)

## Project Structure

| Path | Description |
|------|-------------|
| [configs/](configs/) | Benchmark configurations |
| [docker/](docker/) | Docker configurations |
| [scripts/](scripts/) | Automation scripts |
| [src/](src/) | Source code |
| [video/](video/) | Test videos (*.mp4) |

### Key Files

| File | Description |
|------|-------------|
| [src/benchmark.py](src/benchmark.py) | Main benchmark runner |
| [src/gpu_monitor.py](src/gpu_monitor.py) | GPU metrics collection |
| [src/video_loader.py](src/video_loader.py) | Video frame extraction |
| [scripts/run_all_benchmarks.sh](scripts/run_all_benchmarks.sh) | Run full benchmark suite |
| [scripts/cleanup_gpu.sh](scripts/cleanup_gpu.sh) | Stop containers, free GPU RAM |

## Quick Start

### Requirements

- NVIDIA GPU with CUDA ([tested on RTX A6000](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/))
- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Python 3.10+](https://www.python.org/downloads/)

### Install

```bash
pip install -r requirements.txt
```

### Run All Benchmarks

```bash
./scripts/run_all_benchmarks.sh
```

### Run Single Backend

```bash
# vLLM
docker build -t docker-vllm-awq -f docker/vllm-awq/Dockerfile docker/vllm-awq/
docker run -d --gpus all --name vlm-bench-vllm-awq -p 28001:28001 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-vllm-awq

python src/benchmark.py --config configs/benchmark_vllm.yaml

docker stop vlm-bench-vllm-awq && docker rm vlm-bench-vllm-awq
```

## Methodology

### Benchmark Process

1. **Warmup** - 1 inference run to populate caches (excluded from results)
2. **Benchmark** - 5 runs per video, metrics averaged
3. **Metrics** - Latency (p50/p90/p99), throughput (tokens/s), GPU memory

### Test Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Videos | 5 | Quick iteration while statistically meaningful |
| Frames/video | 4 | Balance between context and speed |
| Temperature | 0.0 | Deterministic outputs for reproducibility |
| Max tokens | 512 | Sufficient for video descriptions |

### Hardware

- **GPU**: NVIDIA RTX A6000 (48 GB VRAM)
- **CUDA**: 12.1
- **Driver**: 535.x

### Why These Backends?

| Backend | Why Included |
|---------|--------------|
| [vLLM](https://docs.vllm.ai/) | Industry standard, best throughput with [PagedAttention](https://arxiv.org/abs/2309.06180) |
| [SGLang](https://github.com/sgl-project/sglang) | Emerging alternative with [RadixAttention](https://arxiv.org/abs/2312.07104) |
| [Ollama](https://ollama.ai/) | Lightweight option using [GGUF quantization](https://github.com/ggerganov/ggml) |

## Backends

| Backend | Port | Config | Memory |
|---------|------|--------|--------|
| [vLLM](docker/vllm-awq/) | 28001 | [benchmark_vllm.yaml](configs/benchmark_vllm.yaml) | ~43 GB |
| [SGLang](docker/sglang/) | 28004 | [benchmark_sglang.yaml](configs/benchmark_sglang.yaml) | ~40 GB |
| [Ollama](docker/ollama/) | 28434 | [benchmark_ollama.yaml](configs/benchmark_ollama.yaml) | ~7 GB |

### Thinking Mode

Add `/think` prefix for reasoning mode. See configs:
- [benchmark_vllm_thinking.yaml](configs/benchmark_vllm_thinking.yaml)
- [benchmark_sglang_thinking.yaml](configs/benchmark_sglang_thinking.yaml)
- [benchmark_ollama_thinking.yaml](configs/benchmark_ollama_thinking.yaml)

## Configuration

Example: [configs/benchmark_vllm.yaml](configs/benchmark_vllm.yaml)

```yaml
backend:
  type: vllm
  base_url: http://localhost:28001
  model: Qwen/Qwen3-VL-4B-Instruct

benchmark:
  video_dir: video
  num_videos: 5
  frames_per_video: 4
```

See [configs/backends/](configs/backends/) for backend-specific options.

## Results

Output: `results/*.json` and `results/*.csv`

### Metrics

| Metric | Description |
|--------|-------------|
| `latency_mean_ms` | Average inference latency |
| `latency_p50_ms` | Median latency |
| `tokens_per_second` | Generation throughput |
| `peak_memory_mb` | Peak GPU memory |

### Sample Results (RTX A6000)

| Backend | Latency (p50) | Tokens/s | Memory |
|---------|---------------|----------|--------|
| vLLM (warm) | 2,787 ms | 59.5 | 43.4 GB |
| SGLang | 3,100 ms | 55.2 | 40.1 GB |
| Ollama | 5,093 ms | 50.3 | 7.1 GB |

### Key Findings

**vLLM is fastest** (~30% faster than SGLang)
- [Prefix caching](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html) reuses KV cache across requests
- [Chunked prefill](https://docs.vllm.ai/en/latest/models/performance.html) overlaps computation with memory ops
- Warm runs 36% faster than cold (2,787ms vs 4,339ms)

**Ollama is most memory efficient** (6x less than vLLM)
- [GGUF Q4_K_M quantization](https://huggingface.co/docs/hub/gguf) reduces 4B params to ~2.5 GB
- Tradeoff: 2x slower than vLLM

**Recommendations**

| Use Case | Backend | Why |
|----------|---------|-----|
| Production (speed) | vLLM | Lowest latency, best throughput |
| Memory constrained | Ollama | 7 GB vs 43 GB, runs on consumer GPUs |
| Development | SGLang | Good balance, easy debugging |

Compare results: `python src/compare.py results/*.json`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | `sudo systemctl stop ollama` |
| GPU OOM | Run [cleanup_gpu.sh](scripts/cleanup_gpu.sh) |
| Container logs | `docker logs vlm-bench-vllm-awq` |

## License

MIT
