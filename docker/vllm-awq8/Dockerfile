# docker/vllm-awq8/Dockerfile
# vLLM with AWQ 8-bit quantization

FROM vllm/vllm-openai:latest

EXPOSE 28003

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:28003/v1/models || exit 1

ENTRYPOINT ["vllm", "serve"]
CMD ["cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit", \
     "--dtype", "auto", \
     "--quantization", "awq", \
     "--gpu-memory-utilization", "0.75", \
     "--max-model-len", "16384", \
     "--port", "28003", \
     "--host", "0.0.0.0"]
