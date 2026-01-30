# docker/sglang/Dockerfile
# SGLang for Qwen3-VL-4B

FROM lmsysorg/sglang:latest

EXPOSE 8004

CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "Qwen/Qwen3-VL-4B-Instruct", \
     "--dtype", "bfloat16", \
     "--mem-fraction-static", "0.85", \
     "--port", "8004", \
     "--host", "0.0.0.0"]
