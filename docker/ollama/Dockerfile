# docker/ollama/Dockerfile
# Ollama for Qwen3-VL-4B

FROM ollama/ollama:latest

EXPOSE 28434

ENV OLLAMA_HOST=0.0.0.0:28434

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:28434/api/tags || exit 1

# Model will be pulled at runtime:
# docker exec -it <container> ollama pull qwen3-vl:4b
