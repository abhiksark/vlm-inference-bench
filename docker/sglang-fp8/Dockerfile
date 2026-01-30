# docker/sglang-fp8/Dockerfile
# SGLang with FP8 quantization (W8A16 on Ampere GPUs)

FROM lmsysorg/sglang:latest

EXPOSE 28005

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:28005/v1/models || exit 1

CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "Qwen/Qwen3-VL-4B-Instruct-FP8", \
     "--dtype", "auto", \
     "--mem-fraction-static", "0.75", \
     "--port", "28005", \
     "--host", "0.0.0.0"]
