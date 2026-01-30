# src/compare.py
"""Multi-backend comparison runner for VLM benchmarking."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import BenchmarkResults, VLMBenchmark


@dataclass
class ComparisonResults:
    """Results from comparing multiple backends."""

    name: str
    timestamp: str
    num_backends: int
    video_dir: str
    num_videos: int
    frames_per_video: int

    backend_results: list[dict] = field(default_factory=list)

    # Rankings (backend name -> rank)
    latency_ranking: list[str] = field(default_factory=list)
    throughput_ranking: list[str] = field(default_factory=list)
    memory_ranking: list[str] = field(default_factory=list)


def load_comparison_config(config_path: str) -> dict:
    """Load comparison suite configuration.

    Args:
        config_path: Path to comparison config YAML.

    Returns:
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_backend(
    backend_config_path: str,
    data_config: dict,
    benchmark_config: dict,
    output_dir: str,
) -> Optional[BenchmarkResults]:
    """Run benchmark for a single backend.

    Args:
        backend_config_path: Path to backend config file.
        data_config: Data configuration (video_dir, etc.).
        benchmark_config: Benchmark configuration (warmup, num_runs).
        output_dir: Directory to save results.

    Returns:
        BenchmarkResults or None if failed.
    """
    # Load backend config
    with open(backend_config_path) as f:
        backend_cfg = yaml.safe_load(f)

    # Merge with data and benchmark config
    backend_cfg["data"] = data_config
    backend_cfg["benchmark"] = benchmark_config

    # Write merged config to temp file
    temp_config_path = Path(output_dir) / "temp_config.yaml"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, "w") as f:
        yaml.dump(backend_cfg, f)

    try:
        benchmark = VLMBenchmark(str(temp_config_path))
        results = benchmark.run_benchmark()
        benchmark.print_summary(results)
        benchmark.save_results(results, output_dir)
        return results
    except Exception as e:
        print(f"Error benchmarking {backend_config_path}: {e}")
        return None
    finally:
        if temp_config_path.exists():
            temp_config_path.unlink()


def run_comparison(config_path: str, output_dir: str) -> ComparisonResults:
    """Run comparison across all backends.

    Args:
        config_path: Path to comparison suite config.
        output_dir: Directory to save results.

    Returns:
        ComparisonResults with all backend results.
    """
    config = load_comparison_config(config_path)

    comparison_name = config.get("name", "VLM Backend Comparison")
    backends = config.get("backends", [])
    data_config = config.get("data", {})
    benchmark_config = config.get("benchmark", {})

    print("=" * 70)
    print(f"COMPARISON SUITE: {comparison_name}")
    print(f"Backends to test: {len(backends)}")
    print("=" * 70)

    all_results: list[BenchmarkResults] = []

    for i, backend_entry in enumerate(backends):
        backend_name = backend_entry.get("name", f"Backend {i+1}")
        backend_config_path = backend_entry.get("config")

        print(f"\n[{i+1}/{len(backends)}] Running: {backend_name}")
        print("-" * 50)

        if not Path(backend_config_path).exists():
            print(f"  Config not found: {backend_config_path}, skipping...")
            continue

        result = run_single_backend(
            backend_config_path,
            data_config,
            benchmark_config,
            output_dir,
        )

        if result:
            # Add friendly name
            result.backend_info["display_name"] = backend_name
            all_results.append(result)

    # Compute rankings
    if all_results:
        # Sort by latency (lower is better)
        latency_ranking = sorted(
            all_results, key=lambda r: r.latency_mean_ms
        )
        # Sort by throughput (higher is better)
        throughput_ranking = sorted(
            all_results, key=lambda r: r.videos_per_second, reverse=True
        )
        # Sort by memory (lower is better)
        memory_ranking = sorted(
            all_results, key=lambda r: r.peak_memory_mb
        )

        latency_names = [
            r.backend_info.get("display_name", r.backend_type)
            for r in latency_ranking
        ]
        throughput_names = [
            r.backend_info.get("display_name", r.backend_type)
            for r in throughput_ranking
        ]
        memory_names = [
            r.backend_info.get("display_name", r.backend_type)
            for r in memory_ranking
        ]
    else:
        latency_names = []
        throughput_names = []
        memory_names = []

    comparison = ComparisonResults(
        name=comparison_name,
        timestamp=datetime.now().isoformat(),
        num_backends=len(all_results),
        video_dir=data_config.get("video_dir", ""),
        num_videos=data_config.get("sample_size", 0),
        frames_per_video=data_config.get("frames_per_video", 8),
        backend_results=[asdict(r) for r in all_results],
        latency_ranking=latency_names,
        throughput_ranking=throughput_names,
        memory_ranking=memory_names,
    )

    return comparison


def print_comparison_summary(comparison: ComparisonResults) -> None:
    """Print comparison summary table.

    Args:
        comparison: ComparisonResults to display.
    """
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)

    # Header
    print(
        f"{'Backend':<30} {'Latency (ms)':<15} {'Videos/sec':<12} "
        f"{'Tokens/sec':<12} {'Memory (MB)':<12}"
    )
    print("-" * 90)

    for result in comparison.backend_results:
        name = result.get("backend_info", {}).get(
            "display_name", result.get("backend_type", "unknown")
        )
        latency = result.get("latency_mean_ms", 0)
        vps = result.get("videos_per_second", 0)
        tps = result.get("tokens_per_second", 0)
        memory = result.get("peak_memory_mb", 0)

        print(f"{name:<30} {latency:<15.2f} {vps:<12.4f} {tps:<12.1f} {memory:<12.0f}")

    print("-" * 90)

    # Rankings
    print("\nRANKINGS:")
    print(f"  Latency (fastest):    {' > '.join(comparison.latency_ranking[:3])}")
    print(f"  Throughput (highest): {' > '.join(comparison.throughput_ranking[:3])}")
    print(f"  Memory (lowest):      {' > '.join(comparison.memory_ranking[:3])}")
    print("=" * 90)


def save_comparison(comparison: ComparisonResults, output_dir: str) -> str:
    """Save comparison results.

    Args:
        comparison: ComparisonResults to save.
        output_dir: Directory to save results.

    Returns:
        Path to saved JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"comparison_{timestamp}.json"

    with open(json_path, "w") as f:
        json.dump(asdict(comparison), f, indent=2)

    print(f"\nComparison results saved to: {json_path}")
    return str(json_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VLM benchmark comparison across multiple backends"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/comparison_suite.yaml",
        help="Path to comparison suite config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    comparison = run_comparison(args.config, args.output_dir)
    print_comparison_summary(comparison)
    save_comparison(comparison, args.output_dir)


if __name__ == "__main__":
    main()
