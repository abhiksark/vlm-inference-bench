# src/backends/vllm_backend.py
"""vLLM backend for VLM inference via OpenAI-compatible API."""

from typing import Any

from . import register_backend
from .openai_compatible import OpenAICompatibleBackend


@register_backend("vllm")
class VLLMBackend(OpenAICompatibleBackend):
    """Backend for vLLM inference server.

    vLLM provides OpenAI-compatible API at /v1/chat/completions.
    Supports various optimizations: flash attention, KV cache, quantization.
    """

    def __init__(self, config):
        """Initialize vLLM backend.

        Args:
            config: Backend configuration with base_url (default: localhost:8000).
        """
        if config.base_url is None:
            config.base_url = "http://localhost:8000"
        super().__init__(config)

    def get_backend_info(self) -> dict[str, Any]:
        """Get vLLM server information.

        Returns:
            Dictionary with vLLM-specific details.
        """
        info = super().get_backend_info()
        info["backend"] = "vllm"

        # Try to get vLLM-specific info
        vllm_options = self.config.extra_options
        if vllm_options:
            info["vllm_options"] = {
                "attention": vllm_options.get("attention", "default"),
                "quantization": vllm_options.get("quantization"),
                "tensor_parallel": vllm_options.get("tensor_parallel_size", 1),
                "gpu_memory_utilization": vllm_options.get(
                    "gpu_memory_utilization", 0.9
                ),
                "prefix_caching": vllm_options.get("enable_prefix_caching", False),
                "chunked_prefill": vllm_options.get("enable_chunked_prefill", False),
            }

        return info
