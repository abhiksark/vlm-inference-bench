# src/backends/__init__.py
"""Backend registry and factory for VLM inference."""

from typing import Type

from .base import BackendConfig, InferenceResponse, VLMBackend

# Registry of available backends
_BACKEND_REGISTRY: dict[str, Type[VLMBackend]] = {}


def register_backend(name: str):
    """Decorator to register a backend class.

    Args:
        name: Backend identifier (e.g., "vllm", "ollama").

    Returns:
        Decorator function.
    """

    def decorator(cls: Type[VLMBackend]) -> Type[VLMBackend]:
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(config: dict) -> VLMBackend:
    """Factory function to create a backend instance.

    Args:
        config: Configuration dictionary with 'backend' section.

    Returns:
        Initialized VLMBackend instance.

    Raises:
        ValueError: If backend type is not registered.
    """
    backend_config = config.get("backend", {})
    backend_type = backend_config.get("type", "transformers")

    if backend_type not in _BACKEND_REGISTRY:
        available = ", ".join(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown backend type: {backend_type}. Available: {available}"
        )

    # Build BackendConfig from dict
    cfg = BackendConfig(
        backend_type=backend_type,
        model_name=backend_config.get("model", config.get("model", {}).get("name", "")),
        base_url=backend_config.get("base_url"),
        device=backend_config.get("device", config.get("model", {}).get("device", "cuda")),
        dtype=backend_config.get("dtype", config.get("model", {}).get("dtype", "bfloat16")),
        max_new_tokens=backend_config.get(
            "max_new_tokens", config.get("model", {}).get("max_new_tokens", 256)
        ),
        timeout=backend_config.get("timeout", 120.0),
        extra_options=backend_config.get("options", {}),
    )

    backend_cls = _BACKEND_REGISTRY[backend_type]
    return backend_cls(cfg)


def list_backends() -> list[str]:
    """List all registered backend types.

    Returns:
        List of backend type names.
    """
    return list(_BACKEND_REGISTRY.keys())


# Import backends to trigger registration
from .transformers_backend import TransformersBackend  # noqa: E402, F401
from .openai_compatible import OpenAICompatibleBackend  # noqa: E402, F401
from .vllm_backend import VLLMBackend  # noqa: E402, F401
from .ollama_backend import OllamaBackend  # noqa: E402, F401
from .tgi_backend import TGIBackend  # noqa: E402, F401
from .sglang_backend import SGLangBackend  # noqa: E402, F401

__all__ = [
    "VLMBackend",
    "BackendConfig",
    "InferenceResponse",
    "get_backend",
    "list_backends",
    "register_backend",
]
