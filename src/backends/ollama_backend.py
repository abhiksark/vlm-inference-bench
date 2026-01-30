# src/backends/ollama_backend.py
"""Ollama backend for VLM inference."""

import base64
import io
import time
from typing import Any, Optional

import requests
from PIL import Image

from . import register_backend
from .base import BackendConfig, InferenceResponse, VLMBackend


@register_backend("ollama")
class OllamaBackend(VLMBackend):
    """Backend for Ollama inference server.

    Ollama uses a custom REST API at /api/generate or /api/chat.
    """

    def __init__(self, config: BackendConfig):
        """Initialize Ollama backend.

        Args:
            config: Backend configuration with base_url (default: localhost:11434).
        """
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.tags_endpoint = f"{self.base_url}/api/tags"
        self.session: Optional[requests.Session] = None

    def initialize(self) -> None:
        """Initialize HTTP session and verify connection."""
        self.session = requests.Session()

        if not self.health_check():
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}"
            )

        self._initialized = True

    def cleanup(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()
            self.session = None
        self._initialized = False

    def health_check(self) -> bool:
        """Check if Ollama server is healthy.

        Returns:
            True if server responds.
        """
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string.

        Args:
            image: PIL Image to encode.

        Returns:
            Base64 encoded string (no data URL prefix).
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def run_inference(
        self, frames: list[Image.Image], prompt: str
    ) -> InferenceResponse:
        """Run inference via Ollama API.

        Args:
            frames: List of PIL Images.
            prompt: Text prompt.

        Returns:
            InferenceResponse with output and metrics.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Encode images
        images_b64 = [self._encode_image_base64(frame) for frame in frames]

        # Use chat endpoint for multi-turn
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ]

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.config.max_new_tokens,
                "temperature": 0,
            },
        }

        start_time = time.perf_counter()

        response = self.session.post(
            self.chat_endpoint,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = response.json()

        output_text = result.get("message", {}).get("content", "")

        # Ollama provides eval_count for tokens
        tokens_generated = result.get("eval_count", len(output_text.split()))

        return InferenceResponse(
            output_text=output_text,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            metadata={
                "model": result.get("model"),
                "prompt_eval_count": result.get("prompt_eval_count"),
                "eval_count": result.get("eval_count"),
                "total_duration_ns": result.get("total_duration"),
                "load_duration_ns": result.get("load_duration"),
                "eval_duration_ns": result.get("eval_duration"),
            },
        )

    def get_backend_info(self) -> dict[str, Any]:
        """Get Ollama server information.

        Returns:
            Dictionary with server details.
        """
        info = {
            "type": "ollama",
            "base_url": self.base_url,
            "model": self.config.model_name,
        }

        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                info["available_models"] = [
                    m["name"] for m in models_data.get("models", [])
                ]
        except requests.RequestException:
            info["available_models"] = []

        return info
