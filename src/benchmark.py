# src/benchmark.py
"""Main VLM benchmarking script with multi-backend support."""

import argparse
import gc
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backends import get_backend, list_backends
from backends.base import VLMBackend
from gpu_monitor import GPUMonitor
from video_loader import VideoLoader


@dataclass
class InferenceResult:
    """Single inference result."""

    video_name: str
    num_frames: int
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    memory_peak_mb: float
    memory_avg_mb: float


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    model_name: str
    backend_type: str
    device: str
    dtype: str
    num_videos: int
    total_frames: int
    warmup_runs: int
    benchmark_runs: int

    # Latency metrics (milliseconds)
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Throughput metrics
    videos_per_second: float = 0.0
    frames_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    model_memory_mb: float = 0.0

    # GPU metrics
    gpu_utilization_avg: float = 0.0
    gpu_utilization_peak: float = 0.0

    # Metadata
    timestamp: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    backend_info: dict = field(default_factory=dict)

    individual_results: list = field(default_factory=list)


class VLMBenchmark:
    """Benchmark runner for Vision Language Models with multi-backend support."""

    def __init__(self, config_path: str):
        """Initialize benchmark with configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.backend: Optional[VLMBackend] = None
        self.gpu_monitor = GPUMonitor()

    def _get_backend_type(self) -> str:
        """Get backend type from config."""
        return self.config.get("backend", {}).get(
            "type", self.config.get("model", {}).get("backend", "transformers")
        )

    def load_backend(self) -> None:
        """Load and initialize the backend."""
        backend_type = self._get_backend_type()
        print(f"Initializing backend: {backend_type}")

        self.backend = get_backend(self.config)
        self.backend.initialize()

        backend_info = self.backend.get_backend_info()
        print(f"Backend ready: {backend_info.get('model', 'unknown')}")

    def run_warmup(self, video_loader: VideoLoader) -> None:
        """Run warmup iterations.

        Args:
            video_loader: VideoLoader instance.
        """
        warmup_runs = self.config.get("benchmark", {}).get("warmup_runs", 3)
        print(f"Running {warmup_runs} warmup iterations...")

        warmup_frames = None
        for _, frames, _ in video_loader.iterate_videos():
            warmup_frames = frames
            break

        if warmup_frames:
            self.backend.warmup(warmup_frames, warmup_runs)

        gc.collect()

    def benchmark_single_video(
        self, frames: list[Image.Image], video_name: str, prompt: str
    ) -> InferenceResult:
        """Benchmark single video inference.

        Args:
            frames: List of PIL Images.
            video_name: Name of the video file.
            prompt: Text prompt.

        Returns:
            InferenceResult with metrics.
        """
        self.gpu_monitor.start()
        response = self.backend.run_inference(frames, prompt)
        gpu_metrics = self.gpu_monitor.stop()

        tokens_per_second = (
            (response.tokens_generated / response.latency_ms) * 1000
            if response.latency_ms > 0
            else 0
        )

        return InferenceResult(
            video_name=video_name,
            num_frames=len(frames),
            latency_ms=response.latency_ms,
            tokens_generated=response.tokens_generated,
            tokens_per_second=tokens_per_second,
            memory_peak_mb=gpu_metrics.peak_memory_mb,
            memory_avg_mb=gpu_metrics.avg_memory_mb,
        )

    def run_benchmark(self, video_dir: Optional[str] = None) -> BenchmarkResults:
        """Run full benchmark suite.

        Args:
            video_dir: Optional override for video directory.

        Returns:
            BenchmarkResults with all metrics.
        """
        data_config = self.config.get("data", {})
        video_dir = video_dir or data_config.get("video_dir", "data/motion")
        sample_size = data_config.get("sample_size")

        video_loader = VideoLoader(
            video_dir=video_dir,
            frames_per_video=data_config.get("frames_per_video", 8),
            sampling_strategy=data_config.get("frame_sampling", "uniform"),
            target_fps=data_config.get("target_fps", 1),
        )

        print(f"Found {video_loader.num_videos} videos in {video_dir}")

        # Load backend
        self.load_backend()
        backend_info = self.backend.get_backend_info()

        # Warmup
        self.run_warmup(video_loader)

        # Run benchmark
        benchmark_config = self.config.get("benchmark", {})
        num_runs = benchmark_config.get("num_runs", 10)
        prompt_prefix = benchmark_config.get("prompt_prefix", "")
        base_prompt = "Describe what is happening in this video. Focus on any motion or activity."
        prompt = f"{prompt_prefix}{base_prompt}"

        results: list[InferenceResult] = []
        total_frames = 0

        videos_to_process = list(video_loader.iterate_videos(sample_size))
        actual_runs = min(num_runs, len(videos_to_process))

        print(f"Running benchmark on {actual_runs} videos...")

        for i in tqdm(range(actual_runs), desc="Benchmarking"):
            video_path, frames, metadata = videos_to_process[i % len(videos_to_process)]

            result = self.benchmark_single_video(frames, metadata["filename"], prompt)
            results.append(result)
            total_frames += len(frames)

        # Aggregate results
        latencies = [r.latency_ms for r in results]
        sorted_latencies = sorted(latencies)

        gpu_info = self.gpu_monitor.get_device_info()

        # Get model config
        model_config = self.config.get("model", {})
        backend_config = self.config.get("backend", {})

        benchmark_results = BenchmarkResults(
            model_name=backend_config.get("model", model_config.get("name", "unknown")),
            backend_type=self._get_backend_type(),
            device=backend_config.get("device", model_config.get("device", "cuda")),
            dtype=backend_config.get("dtype", model_config.get("dtype", "bfloat16")),
            num_videos=actual_runs,
            total_frames=total_frames,
            warmup_runs=benchmark_config.get("warmup_runs", 3),
            benchmark_runs=num_runs,
            latency_mean_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_p50_ms=sorted_latencies[len(sorted_latencies) // 2],
            latency_p90_ms=sorted_latencies[int(len(sorted_latencies) * 0.9)],
            latency_p99_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
            videos_per_second=1000.0 / statistics.mean(latencies),
            frames_per_second=(total_frames / actual_runs)
            * (1000.0 / statistics.mean(latencies)),
            tokens_per_second=statistics.mean([r.tokens_per_second for r in results]),
            peak_memory_mb=max(r.memory_peak_mb for r in results),
            avg_memory_mb=statistics.mean([r.memory_avg_mb for r in results]),
            model_memory_mb=backend_info.get("model_memory_mb", 0),
            gpu_name=gpu_info.get("device_name", "unknown"),
            gpu_memory_gb=gpu_info.get("total_memory_gb", 0),
            timestamp=datetime.now().isoformat(),
            backend_info=backend_info,
            individual_results=[asdict(r) for r in results],
        )

        return benchmark_results

    def save_results(self, results: BenchmarkResults, output_dir: str) -> str:
        """Save benchmark results to files.

        Args:
            results: BenchmarkResults to save.
            output_dir: Directory to save results.

        Returns:
            Path to saved JSON file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = results.model_name.split("/")[-1]
        backend = results.backend_type

        # Save JSON
        json_path = output_path / f"benchmark_{backend}_{model_short}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"Results saved to {json_path}")

        # Save summary CSV
        csv_path = output_path / f"benchmark_{backend}_{model_short}_{timestamp}.csv"
        with open(csv_path, "w") as f:
            f.write("metric,value\n")
            f.write(f"model,{results.model_name}\n")
            f.write(f"backend,{results.backend_type}\n")
            f.write(f"latency_mean_ms,{results.latency_mean_ms:.2f}\n")
            f.write(f"latency_p50_ms,{results.latency_p50_ms:.2f}\n")
            f.write(f"latency_p90_ms,{results.latency_p90_ms:.2f}\n")
            f.write(f"latency_p99_ms,{results.latency_p99_ms:.2f}\n")
            f.write(f"videos_per_second,{results.videos_per_second:.4f}\n")
            f.write(f"frames_per_second,{results.frames_per_second:.2f}\n")
            f.write(f"tokens_per_second,{results.tokens_per_second:.2f}\n")
            f.write(f"peak_memory_mb,{results.peak_memory_mb:.0f}\n")
        print(f"CSV saved to {csv_path}")

        return str(json_path)

    def print_summary(self, results: BenchmarkResults) -> None:
        """Print benchmark summary to console.

        Args:
            results: BenchmarkResults to display.
        """
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Model: {results.model_name}")
        print(f"Backend: {results.backend_type}")
        print(f"Device: {results.gpu_name} ({results.gpu_memory_gb:.1f} GB)")
        print(f"Dtype: {results.dtype}")
        print(f"Videos: {results.num_videos} | Frames: {results.total_frames}")
        print("-" * 60)
        print("LATENCY (ms)")
        print(f"  Mean:   {results.latency_mean_ms:>10.2f}")
        print(f"  Std:    {results.latency_std_ms:>10.2f}")
        print(f"  Min:    {results.latency_min_ms:>10.2f}")
        print(f"  Max:    {results.latency_max_ms:>10.2f}")
        print(f"  P50:    {results.latency_p50_ms:>10.2f}")
        print(f"  P90:    {results.latency_p90_ms:>10.2f}")
        print(f"  P99:    {results.latency_p99_ms:>10.2f}")
        print("-" * 60)
        print("THROUGHPUT")
        print(f"  Videos/sec:  {results.videos_per_second:>10.4f}")
        print(f"  Frames/sec:  {results.frames_per_second:>10.2f}")
        print(f"  Tokens/sec:  {results.tokens_per_second:>10.2f}")
        print("-" * 60)
        print("MEMORY (MB)")
        print(f"  Peak:   {results.peak_memory_mb:>10.0f}")
        print(f"  Avg:    {results.avg_memory_mb:>10.0f}")
        print("=" * 60)

    def cleanup(self) -> None:
        """Clean up backend resources."""
        if self.backend:
            self.backend.cleanup()
            self.backend = None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VLM Benchmark Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Override video directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit",
    )
    args = parser.parse_args()

    if args.list_backends:
        print("Available backends:")
        for backend in list_backends():
            print(f"  - {backend}")
        return

    benchmark = VLMBenchmark(args.config)
    try:
        results = benchmark.run_benchmark(video_dir=args.video_dir)
        benchmark.print_summary(results)
        benchmark.save_results(results, args.output_dir)
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
