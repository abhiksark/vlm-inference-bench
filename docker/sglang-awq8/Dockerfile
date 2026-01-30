# docker/sglang-awq8/Dockerfile
# SGLang with AWQ 8-bit quantization

FROM lmsysorg/sglang:latest

EXPOSE 28006

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:28006/v1/models || exit 1

CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-8bit", \
     "--dtype", "auto", \
     "--mem-fraction-static", "0.75", \
     "--port", "28006", \
     "--host", "0.0.0.0"]
