# docker/vllm-fp8/Dockerfile
# vLLM with FP8 quantization (W8A16 on Ampere GPUs)

FROM vllm/vllm-openai:latest

EXPOSE 28002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:28002/v1/models || exit 1

ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen3-VL-4B-Instruct-FP8", \
     "--dtype", "auto", \
     "--gpu-memory-utilization", "0.85", \
     "--max-model-len", "16384", \
     "--port", "28002", \
     "--host", "0.0.0.0"]
