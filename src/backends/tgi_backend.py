# src/backends/tgi_backend.py
"""Text Generation Inference (TGI) backend for VLM inference."""

from typing import Any

from . import register_backend
from .openai_compatible import OpenAICompatibleBackend


@register_backend("tgi")
class TGIBackend(OpenAICompatibleBackend):
    """Backend for HuggingFace Text Generation Inference server.

    TGI provides OpenAI-compatible API at /v1/chat/completions.
    """

    def __init__(self, config):
        """Initialize TGI backend.

        Args:
            config: Backend configuration with base_url (default: localhost:8002).
        """
        if config.base_url is None:
            config.base_url = "http://localhost:8002"
        super().__init__(config)

    def get_backend_info(self) -> dict[str, Any]:
        """Get TGI server information.

        Returns:
            Dictionary with TGI-specific details.
        """
        info = super().get_backend_info()
        info["backend"] = "tgi"

        # TGI-specific options
        tgi_options = self.config.extra_options
        if tgi_options:
            info["tgi_options"] = {
                "max_total_tokens": tgi_options.get("max_total_tokens"),
                "max_input_length": tgi_options.get("max_input_length"),
                "quantize": tgi_options.get("quantize"),
                "dtype": tgi_options.get("dtype"),
            }

        # Try to get TGI info endpoint
        try:
            import requests

            info_response = requests.get(f"{self.base_url}/info", timeout=5)
            if info_response.status_code == 200:
                info["server_info"] = info_response.json()
        except Exception:
            pass

        return info
