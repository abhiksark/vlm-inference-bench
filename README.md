# VLM Benchmark Suite

Benchmarking framework for Vision-Language Models (VLMs) with multiple inference backends.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
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
docker run -d --gpus all --name vlm-bench-vllm-awq -p 8001:8001 \
    -v ~/.cache/huggingface:/root/.cache/huggingface docker-vllm-awq

python src/benchmark.py --config configs/benchmark_vllm.yaml

docker stop vlm-bench-vllm-awq && docker rm vlm-bench-vllm-awq
```

## Backends

| Backend | Port | Config | Memory |
|---------|------|--------|--------|
| [vLLM](docker/vllm-awq/) | 8001 | [benchmark_vllm.yaml](configs/benchmark_vllm.yaml) | ~43 GB |
| [SGLang](docker/sglang/) | 8004 | [benchmark_sglang.yaml](configs/benchmark_sglang.yaml) | ~40 GB |
| [Ollama](docker/ollama/) | 11434 | [benchmark_ollama.yaml](configs/benchmark_ollama.yaml) | ~7 GB |

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
  base_url: http://localhost:8001
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

Compare results: `python src/compare.py results/*.json`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port in use | `sudo systemctl stop ollama` |
| GPU OOM | Run [cleanup_gpu.sh](scripts/cleanup_gpu.sh) |
| Container logs | `docker logs vlm-bench-vllm-awq` |

## License

MIT
