# src/backends/openai_compatible.py
"""Base class for OpenAI-compatible VLM backends (vLLM, TGI, SGLang)."""

import base64
import io
import time
from typing import Any, Optional

import requests
from PIL import Image

from .base import BackendConfig, InferenceResponse, VLMBackend


class OpenAICompatibleBackend(VLMBackend):
    """Base class for backends using OpenAI-compatible chat completions API."""

    def __init__(self, config: BackendConfig):
        """Initialize OpenAI-compatible backend.

        Args:
            config: Backend configuration with base_url.
        """
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        self.models_endpoint = f"{self.base_url}/v1/models"
        self.session: Optional[requests.Session] = None

    def initialize(self) -> None:
        """Initialize HTTP session and verify connection."""
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        if not self.health_check():
            raise ConnectionError(
                f"Failed to connect to {self.name} at {self.base_url}"
            )

        self._initialized = True

    def cleanup(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()
            self.session = None
        self._initialized = False

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server responds to models endpoint.
        """
        try:
            response = requests.get(self.models_endpoint, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64 encoded string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_messages(
        self, frames: list[Image.Image], prompt: str
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages with images.

        Args:
            frames: List of PIL Images.
            prompt: Text prompt.

        Returns:
            Messages list for chat completions API.
        """
        content = []

        for frame in frames:
            image_b64 = self._encode_image_base64(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def run_inference(
        self, frames: list[Image.Image], prompt: str
    ) -> InferenceResponse:
        """Run inference via OpenAI-compatible API.

        Args:
            frames: List of PIL Images.
            prompt: Text prompt.

        Returns:
            InferenceResponse with output and metrics.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        messages = self._build_messages(frames, prompt)

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": 0,
            "stream": False,
        }

        start_time = time.perf_counter()

        response = self.session.post(
            self.api_endpoint,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = response.json()

        output_text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        tokens_generated = usage.get("completion_tokens", len(output_text.split()))

        return InferenceResponse(
            output_text=output_text,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            metadata={
                "prompt_tokens": usage.get("prompt_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "model": result.get("model"),
            },
        )

    def get_backend_info(self) -> dict[str, Any]:
        """Get backend server information.

        Returns:
            Dictionary with server details.
        """
        info = {
            "type": self.config.backend_type,
            "base_url": self.base_url,
            "model": self.config.model_name,
        }

        try:
            response = requests.get(self.models_endpoint, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                info["available_models"] = models_data.get("data", [])
        except requests.RequestException:
            info["available_models"] = []

        return info
