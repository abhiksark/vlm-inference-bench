# docker/tgi/Dockerfile
# HuggingFace Text Generation Inference for Qwen3-VL-4B

FROM ghcr.io/huggingface/text-generation-inference:latest

EXPOSE 80

CMD ["--model-id", "Qwen/Qwen3-VL-4B-Instruct", \
     "--dtype", "bfloat16", \
     "--max-total-tokens", "4096", \
     "--max-input-length", "2048"]
