# src/backends/transformers_backend.py
"""Transformers/HuggingFace backend for local VLM inference."""

import gc
import time
from typing import Any

import torch
from PIL import Image

from . import register_backend
from .base import BackendConfig, InferenceResponse, VLMBackend


@register_backend("transformers")
class TransformersBackend(VLMBackend):
    """Backend for local inference using HuggingFace Transformers."""

    def __init__(self, config: BackendConfig):
        """Initialize Transformers backend.

        Args:
            config: Backend configuration.
        """
        super().__init__(config)
        self.model = None
        self.processor = None
        self._dtype = self._get_dtype(config.dtype)

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def initialize(self) -> None:
        """Load the model and processor."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        attn_impl = self.config.extra_options.get(
            "attn_implementation", "flash_attention_2"
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=self._dtype,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

        self._initialized = True

    def cleanup(self) -> None:
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        torch.cuda.empty_cache()
        gc.collect()
        self._initialized = False

    def health_check(self) -> bool:
        """Check if model is loaded and GPU is available.

        Returns:
            True if ready for inference.
        """
        return (
            self._initialized
            and self.model is not None
            and torch.cuda.is_available()
        )

    def _prepare_input(self, frames: list[Image.Image], prompt: str) -> dict:
        """Prepare model input from frames and prompt.

        Args:
            frames: List of PIL Images.
            prompt: Text prompt.

        Returns:
            Model inputs dictionary.
        """
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs.to(self.config.device)

    def run_inference(
        self, frames: list[Image.Image], prompt: str
    ) -> InferenceResponse:
        """Run local inference.

        Args:
            frames: List of PIL Images.
            prompt: Text prompt.

        Returns:
            InferenceResponse with output and metrics.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        inputs = self._prepare_input(frames, prompt)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000

        generated_ids = [
            out_ids[in_len:]
            for in_len, out_ids in zip([input_len], output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        tokens_generated = len(output_ids[0]) - input_len

        return InferenceResponse(
            output_text=output_text,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
            metadata={
                "input_tokens": input_len,
                "device": self.config.device,
                "dtype": self.config.dtype,
            },
        )

    def get_backend_info(self) -> dict[str, Any]:
        """Get backend information.

        Returns:
            Dictionary with model and device info.
        """
        info = {
            "type": "transformers",
            "model": self.config.model_name,
            "device": self.config.device,
            "dtype": self.config.dtype,
        }

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )

        if self._initialized and self.model is not None:
            info["model_memory_mb"] = torch.cuda.memory_allocated() / (1024**2)

        return info

    def warmup(self, frames: list[Image.Image], num_runs: int = 3) -> None:
        """Run warmup with cache clearing.

        Args:
            frames: Sample frames.
            num_runs: Number of warmup runs.
        """
        super().warmup(frames, num_runs)
        torch.cuda.empty_cache()
        gc.collect()
