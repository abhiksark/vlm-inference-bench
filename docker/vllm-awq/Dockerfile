# docker/vllm-awq/Dockerfile
# vLLM for Qwen3-VL-4B (optimized baseline)
# Note: vLLM v0.14.1 enables prefix-caching and chunked-prefill by default

FROM vllm/vllm-openai:latest

EXPOSE 28001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:28001/v1/models || exit 1

ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen3-VL-4B-Instruct", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.85", \
     "--max-model-len", "16384", \
     "--port", "28001", \
     "--host", "0.0.0.0"]
