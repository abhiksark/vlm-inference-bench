# src/backends/base.py
"""Abstract base class for VLM inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image


@dataclass
class InferenceResponse:
    """Response from a single inference call."""

    output_text: str
    latency_ms: float
    tokens_generated: int
    time_to_first_token_ms: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BackendConfig:
    """Configuration for a VLM backend."""

    backend_type: str
    model_name: str
    base_url: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    timeout: float = 120.0
    extra_options: dict = field(default_factory=dict)


class VLMBackend(ABC):
    """Abstract base class for VLM inference backends."""

    def __init__(self, config: BackendConfig):
        """Initialize backend with configuration.

        Args:
            config: Backend configuration.
        """
        self.config = config
        self._initialized = False

    @property
    def name(self) -> str:
        """Return backend name."""
        return self.config.backend_type

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.config.model_name

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load model, connect to server, etc.)."""
        pass

    @abstractmethod
    def run_inference(
        self, frames: list[Image.Image], prompt: str
    ) -> InferenceResponse:
        """Run inference on frames with prompt.

        Args:
            frames: List of PIL Images (video frames).
            prompt: Text prompt for the model.

        Returns:
            InferenceResponse with output and metrics.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (unload model, close connections, etc.)."""
        pass

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the backend.

        Returns:
            Dictionary with backend details (version, capabilities, etc.).
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is healthy and ready.

        Returns:
            True if backend is ready for inference.
        """
        pass

    def warmup(self, frames: list[Image.Image], num_runs: int = 3) -> None:
        """Run warmup inferences.

        Args:
            frames: Sample frames for warmup.
            num_runs: Number of warmup iterations.
        """
        prompt = "Describe this image briefly."
        for _ in range(num_runs):
            self.run_inference(frames, prompt)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
