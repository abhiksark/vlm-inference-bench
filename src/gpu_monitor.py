# src/gpu_monitor.py
"""GPU and memory monitoring utilities for benchmarking."""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GPUMetrics:
    """Container for GPU metrics."""

    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    peak_utilization: float = 0.0
    avg_utilization: float = 0.0
    memory_samples: list = field(default_factory=list)
    utilization_samples: list = field(default_factory=list)


class GPUMonitor:
    """Monitor GPU memory and utilization during inference."""

    def __init__(self, device_id: int = 0, sample_interval: float = 0.1):
        """Initialize GPU monitor.

        Args:
            device_id: CUDA device ID to monitor.
            sample_interval: Sampling interval in seconds.
        """
        self.device_id = device_id
        self.sample_interval = sample_interval
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._metrics = GPUMetrics()
        self._pynvml_available = False

        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self._pynvml_available = True
            self._pynvml = pynvml
        except Exception:
            pass

    def _sample_metrics(self) -> None:
        """Sample GPU metrics in background thread."""
        while self._monitoring:
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    memory_mb = torch.cuda.memory_allocated(self.device_id) / (1024**2)
                elif self._pynvml_available:
                    mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    memory_mb = mem_info.used / (1024**2)
                else:
                    memory_mb = 0.0
                self._metrics.memory_samples.append(memory_mb)
            except Exception:
                pass

            if self._pynvml_available:
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._metrics.utilization_samples.append(util.gpu)
                except Exception:
                    pass

            time.sleep(self.sample_interval)

    def start(self) -> None:
        """Start monitoring GPU metrics."""
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats(self.device_id)
        except Exception:
            pass
        self._metrics = GPUMetrics()
        self._monitoring = True
        self._thread = threading.Thread(target=self._sample_metrics, daemon=True)
        self._thread.start()

    def stop(self) -> GPUMetrics:
        """Stop monitoring and return collected metrics.

        Returns:
            GPUMetrics with collected data.
        """
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)

        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                peak_memory = torch.cuda.max_memory_allocated(self.device_id) / (1024**2)
            elif self._metrics.memory_samples:
                peak_memory = max(self._metrics.memory_samples)
            else:
                peak_memory = 0.0
        except Exception:
            peak_memory = max(self._metrics.memory_samples) if self._metrics.memory_samples else 0.0
        self._metrics.peak_memory_mb = peak_memory

        if self._metrics.memory_samples:
            self._metrics.avg_memory_mb = sum(self._metrics.memory_samples) / len(
                self._metrics.memory_samples
            )

        if self._metrics.utilization_samples:
            self._metrics.peak_utilization = max(self._metrics.utilization_samples)
            self._metrics.avg_utilization = sum(
                self._metrics.utilization_samples
            ) / len(self._metrics.utilization_samples)

        return self._metrics

    def get_device_info(self) -> dict:
        """Get static GPU device information.

        Returns:
            Dictionary with device info.
        """
        info = {}
        try:
            if torch.cuda.is_available():
                info = {
                    "device_name": torch.cuda.get_device_name(self.device_id),
                    "total_memory_gb": torch.cuda.get_device_properties(
                        self.device_id
                    ).total_memory
                    / (1024**3),
                    "compute_capability": torch.cuda.get_device_capability(self.device_id),
                }
        except Exception:
            pass

        if not info and self._pynvml_available:
            try:
                name = self._pynvml.nvmlDeviceGetName(self._handle)
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                info = {
                    "device_name": name if isinstance(name, str) else name.decode(),
                    "total_memory_gb": mem_info.total / (1024**3),
                    "compute_capability": (0, 0),
                }
            except Exception:
                info = {"device_name": "unknown", "total_memory_gb": 0, "compute_capability": (0, 0)}

        return info or {"device_name": "unknown", "total_memory_gb": 0, "compute_capability": (0, 0)}


def get_memory_snapshot() -> dict:
    """Get current memory snapshot.

    Returns:
        Dictionary with current memory stats.
    """
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }
