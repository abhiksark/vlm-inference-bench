# docker/vllm/Dockerfile
# vLLM with Flash Attention for Qwen3-VL-4B

FROM vllm/vllm-openai:latest

EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen3-VL-4B-Instruct", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.85", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
