# docker/sglang/Dockerfile
# SGLang for Qwen3-VL-4B

FROM lmsysorg/sglang:latest

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8004/v1/models || exit 1

CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "Qwen/Qwen3-VL-4B-Instruct", \
     "--dtype", "bfloat16", \
     "--mem-fraction-static", "0.85", \
     "--port", "8004", \
     "--host", "0.0.0.0"]
