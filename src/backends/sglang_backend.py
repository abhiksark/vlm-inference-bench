# src/backends/sglang_backend.py
"""SGLang backend for VLM inference."""

from typing import Any

from . import register_backend
from .openai_compatible import OpenAICompatibleBackend


@register_backend("sglang")
class SGLangBackend(OpenAICompatibleBackend):
    """Backend for SGLang inference server.

    SGLang provides OpenAI-compatible API at /v1/chat/completions.
    Known for efficient batching and RadixAttention.
    """

    def __init__(self, config):
        """Initialize SGLang backend.

        Args:
            config: Backend configuration with base_url (default: localhost:8003).
        """
        if config.base_url is None:
            config.base_url = "http://localhost:8003"
        super().__init__(config)

    def get_backend_info(self) -> dict[str, Any]:
        """Get SGLang server information.

        Returns:
            Dictionary with SGLang-specific details.
        """
        info = super().get_backend_info()
        info["backend"] = "sglang"

        # SGLang-specific options
        sglang_options = self.config.extra_options
        if sglang_options:
            info["sglang_options"] = {
                "mem_fraction": sglang_options.get("mem_fraction"),
                "tp_size": sglang_options.get("tp_size", 1),
                "dp_size": sglang_options.get("dp_size", 1),
                "schedule_policy": sglang_options.get("schedule_policy", "lpm"),
            }

        # Try to get SGLang server info
        try:
            import requests

            # SGLang may have /get_model_info endpoint
            model_info = requests.get(
                f"{self.base_url}/get_model_info", timeout=5
            )
            if model_info.status_code == 200:
                info["model_info"] = model_info.json()
        except Exception:
            pass

        return info
