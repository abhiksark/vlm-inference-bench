# src/visualize.py
"""Visualization tools for benchmark results."""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: str) -> dict:
    """Load benchmark results from JSON file.

    Args:
        results_path: Path to JSON results file.

    Returns:
        Dictionary with benchmark results.
    """
    with open(results_path) as f:
        return json.load(f)


def plot_latency_distribution(results: dict, output_path: Optional[str] = None) -> None:
    """Plot latency distribution histogram.

    Args:
        results: Benchmark results dictionary.
        output_path: Optional path to save figure.
    """
    latencies = [r["latency_ms"] for r in results["individual_results"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(latencies, bins=30, edgecolor="black", alpha=0.7, color="#2196F3")

    # Add percentile lines
    p50 = results["latency_p50_ms"]
    p90 = results["latency_p90_ms"]
    p99 = results["latency_p99_ms"]

    ax.axvline(p50, color="green", linestyle="--", linewidth=2, label=f"P50: {p50:.1f}ms")
    ax.axvline(p90, color="orange", linestyle="--", linewidth=2, label=f"P90: {p90:.1f}ms")
    ax.axvline(p99, color="red", linestyle="--", linewidth=2, label=f"P99: {p99:.1f}ms")

    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Inference Latency Distribution\n{results['model_name']}", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved latency distribution to {output_path}")
    else:
        plt.show()


def plot_memory_timeline(results: dict, output_path: Optional[str] = None) -> None:
    """Plot memory usage across inferences.

    Args:
        results: Benchmark results dictionary.
        output_path: Optional path to save figure.
    """
    individual = results["individual_results"]
    indices = range(len(individual))
    peak_memory = [r["memory_peak_mb"] for r in individual]
    avg_memory = [r["memory_avg_mb"] for r in individual]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(indices, peak_memory, alpha=0.3, color="#F44336", label="Peak Memory")
    ax.plot(indices, peak_memory, color="#F44336", linewidth=1)
    ax.plot(indices, avg_memory, color="#4CAF50", linewidth=2, label="Avg Memory")

    ax.axhline(
        results["model_memory_mb"],
        color="#9C27B0",
        linestyle="--",
        linewidth=2,
        label=f"Model Memory: {results['model_memory_mb']:.0f} MB",
    )

    ax.set_xlabel("Inference #", fontsize=12)
    ax.set_ylabel("Memory (MB)", fontsize=12)
    ax.set_title(f"GPU Memory Usage\n{results['model_name']}", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved memory timeline to {output_path}")
    else:
        plt.show()


def plot_throughput_metrics(results: dict, output_path: Optional[str] = None) -> None:
    """Plot throughput metrics bar chart.

    Args:
        results: Benchmark results dictionary.
        output_path: Optional path to save figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Videos per second
    axes[0].bar(["Videos/sec"], [results["videos_per_second"]], color="#2196F3", width=0.5)
    axes[0].set_ylabel("Videos per Second")
    axes[0].set_title("Video Throughput")
    axes[0].text(
        0,
        results["videos_per_second"] / 2,
        f"{results['videos_per_second']:.3f}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Frames per second
    axes[1].bar(["Frames/sec"], [results["frames_per_second"]], color="#4CAF50", width=0.5)
    axes[1].set_ylabel("Frames per Second")
    axes[1].set_title("Frame Throughput")
    axes[1].text(
        0,
        results["frames_per_second"] / 2,
        f"{results['frames_per_second']:.1f}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Tokens per second
    axes[2].bar(["Tokens/sec"], [results["tokens_per_second"]], color="#FF9800", width=0.5)
    axes[2].set_ylabel("Tokens per Second")
    axes[2].set_title("Token Generation")
    axes[2].text(
        0,
        results["tokens_per_second"] / 2,
        f"{results['tokens_per_second']:.1f}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.suptitle(f"Throughput Metrics - {results['model_name']}", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved throughput metrics to {output_path}")
    else:
        plt.show()


def plot_latency_by_frames(results: dict, output_path: Optional[str] = None) -> None:
    """Plot latency vs number of frames.

    Args:
        results: Benchmark results dictionary.
        output_path: Optional path to save figure.
    """
    individual = results["individual_results"]
    frames = [r["num_frames"] for r in individual]
    latencies = [r["latency_ms"] for r in individual]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(frames, latencies, alpha=0.6, c="#2196F3", s=50)

    # Add trend line
    if len(set(frames)) > 1:
        z = np.polyfit(frames, latencies, 1)
        p = np.poly1d(z)
        frame_range = np.linspace(min(frames), max(frames), 100)
        ax.plot(frame_range, p(frame_range), "r--", linewidth=2, label="Trend")
        ax.legend()

    ax.set_xlabel("Number of Frames", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Latency vs Frame Count\n{results['model_name']}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved latency vs frames to {output_path}")
    else:
        plt.show()


def generate_report(results: dict, output_dir: str) -> None:
    """Generate full benchmark report with all visualizations.

    Args:
        results: Benchmark results dictionary.
        output_dir: Directory to save report files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_short = results["model_name"].split("/")[-1]

    # Generate all plots
    plot_latency_distribution(
        results, str(output_path / f"{model_short}_latency_dist.png")
    )
    plot_memory_timeline(
        results, str(output_path / f"{model_short}_memory_timeline.png")
    )
    plot_throughput_metrics(
        results, str(output_path / f"{model_short}_throughput.png")
    )
    plot_latency_by_frames(
        results, str(output_path / f"{model_short}_latency_frames.png")
    )

    # Generate markdown report
    report_path = output_path / f"{model_short}_report.md"
    with open(report_path, "w") as f:
        f.write(f"# Benchmark Report: {results['model_name']}\n\n")
        f.write(f"**Date:** {results['timestamp']}\n\n")
        f.write("## Hardware\n\n")
        f.write(f"- **GPU:** {results['gpu_name']}\n")
        f.write(f"- **VRAM:** {results['gpu_memory_gb']:.1f} GB\n")
        f.write(f"- **Dtype:** {results['dtype']}\n\n")
        f.write("## Test Configuration\n\n")
        f.write(f"- **Videos:** {results['num_videos']}\n")
        f.write(f"- **Total Frames:** {results['total_frames']}\n")
        f.write(f"- **Warmup Runs:** {results['warmup_runs']}\n\n")
        f.write("## Latency (ms)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {results['latency_mean_ms']:.2f} |\n")
        f.write(f"| Std | {results['latency_std_ms']:.2f} |\n")
        f.write(f"| Min | {results['latency_min_ms']:.2f} |\n")
        f.write(f"| Max | {results['latency_max_ms']:.2f} |\n")
        f.write(f"| P50 | {results['latency_p50_ms']:.2f} |\n")
        f.write(f"| P90 | {results['latency_p90_ms']:.2f} |\n")
        f.write(f"| P99 | {results['latency_p99_ms']:.2f} |\n\n")
        f.write("## Throughput\n\n")
        f.write(f"- **Videos/sec:** {results['videos_per_second']:.4f}\n")
        f.write(f"- **Frames/sec:** {results['frames_per_second']:.2f}\n")
        f.write(f"- **Tokens/sec:** {results['tokens_per_second']:.2f}\n\n")
        f.write("## Memory (MB)\n\n")
        f.write(f"- **Model:** {results['model_memory_mb']:.0f}\n")
        f.write(f"- **Peak:** {results['peak_memory_mb']:.0f}\n")
        f.write(f"- **Average:** {results['avg_memory_mb']:.0f}\n\n")
        f.write("## Visualizations\n\n")
        f.write(f"![Latency Distribution]({model_short}_latency_dist.png)\n\n")
        f.write(f"![Memory Timeline]({model_short}_memory_timeline.png)\n\n")
        f.write(f"![Throughput]({model_short}_throughput.png)\n\n")
        f.write(f"![Latency vs Frames]({model_short}_latency_frames.png)\n")

    print(f"Report saved to {report_path}")


def plot_comparison_latency(
    comparison: dict, output_path: Optional[str] = None
) -> None:
    """Plot latency comparison across backends.

    Args:
        comparison: Comparison results dictionary.
        output_path: Optional path to save figure.
    """
    backends = comparison["backend_results"]
    names = [
        b.get("backend_info", {}).get("display_name", b.get("backend_type", "unknown"))
        for b in backends
    ]

    latency_mean = [b["latency_mean_ms"] for b in backends]
    latency_p50 = [b["latency_p50_ms"] for b in backends]
    latency_p90 = [b["latency_p90_ms"] for b in backends]
    latency_p99 = [b["latency_p99_ms"] for b in backends]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - 1.5 * width, latency_mean, width, label="Mean", color="#2196F3")
    bars2 = ax.bar(x - 0.5 * width, latency_p50, width, label="P50", color="#4CAF50")
    bars3 = ax.bar(x + 0.5 * width, latency_p90, width, label="P90", color="#FF9800")
    bars4 = ax.bar(x + 1.5 * width, latency_p99, width, label="P99", color="#F44336")

    ax.set_xlabel("Backend", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Latency Comparison\n{comparison['name']}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison latency to {output_path}")
    else:
        plt.show()


def plot_comparison_throughput(
    comparison: dict, output_path: Optional[str] = None
) -> None:
    """Plot throughput comparison across backends.

    Args:
        comparison: Comparison results dictionary.
        output_path: Optional path to save figure.
    """
    backends = comparison["backend_results"]
    names = [
        b.get("backend_info", {}).get("display_name", b.get("backend_type", "unknown"))
        for b in backends
    ]

    videos_per_sec = [b["videos_per_second"] for b in backends]
    tokens_per_sec = [b["tokens_per_second"] for b in backends]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Videos per second
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars1 = axes[0].barh(names, videos_per_sec, color=colors)
    axes[0].set_xlabel("Videos per Second")
    axes[0].set_title("Video Throughput")
    axes[0].grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars1, videos_per_sec):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    # Tokens per second
    bars2 = axes[1].barh(names, tokens_per_sec, color=colors)
    axes[1].set_xlabel("Tokens per Second")
    axes[1].set_title("Token Generation")
    axes[1].grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars2, tokens_per_sec):
        axes[1].text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center")

    plt.suptitle(f"Throughput Comparison\n{comparison['name']}", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison throughput to {output_path}")
    else:
        plt.show()


def plot_comparison_memory(
    comparison: dict, output_path: Optional[str] = None
) -> None:
    """Plot memory comparison across backends.

    Args:
        comparison: Comparison results dictionary.
        output_path: Optional path to save figure.
    """
    backends = comparison["backend_results"]
    names = [
        b.get("backend_info", {}).get("display_name", b.get("backend_type", "unknown"))
        for b in backends
    ]

    peak_memory = [b["peak_memory_mb"] for b in backends]
    avg_memory = [b["avg_memory_mb"] for b in backends]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, peak_memory, width, label="Peak Memory", color="#F44336")
    bars2 = ax.bar(x + width / 2, avg_memory, width, label="Avg Memory", color="#4CAF50")

    ax.set_xlabel("Backend", fontsize=12)
    ax.set_ylabel("Memory (MB)", fontsize=12)
    ax.set_title(f"Memory Usage Comparison\n{comparison['name']}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison memory to {output_path}")
    else:
        plt.show()


def plot_comparison_radar(
    comparison: dict, output_path: Optional[str] = None
) -> None:
    """Plot radar chart comparing all metrics across backends.

    Args:
        comparison: Comparison results dictionary.
        output_path: Optional path to save figure.
    """
    backends = comparison["backend_results"]
    names = [
        b.get("backend_info", {}).get("display_name", b.get("backend_type", "unknown"))
        for b in backends
    ]

    # Metrics to compare (normalized to 0-1 scale, higher is better)
    metrics = ["Latency", "Throughput", "Tokens/sec", "Memory Eff."]
    num_metrics = len(metrics)

    # Normalize metrics (invert latency and memory so higher = better)
    latencies = [b["latency_mean_ms"] for b in backends]
    throughputs = [b["videos_per_second"] for b in backends]
    tokens = [b["tokens_per_second"] for b in backends]
    memories = [b["peak_memory_mb"] for b in backends]

    max_lat = max(latencies) if latencies else 1
    max_thr = max(throughputs) if throughputs else 1
    max_tok = max(tokens) if tokens else 1
    max_mem = max(memories) if memories else 1

    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(backends)))

    for i, backend in enumerate(backends):
        values = [
            1 - backend["latency_mean_ms"] / max_lat,  # Invert: lower is better
            backend["videos_per_second"] / max_thr,
            backend["tokens_per_second"] / max_tok,
            1 - backend["peak_memory_mb"] / max_mem,  # Invert: lower is better
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=names[i], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f"Backend Comparison Radar\n{comparison['name']}", fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison radar to {output_path}")
    else:
        plt.show()


def generate_comparison_report(comparison: dict, output_dir: str) -> None:
    """Generate full comparison report with all visualizations.

    Args:
        comparison: Comparison results dictionary.
        output_dir: Directory to save report files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all comparison plots
    plot_comparison_latency(comparison, str(output_path / "comparison_latency.png"))
    plot_comparison_throughput(comparison, str(output_path / "comparison_throughput.png"))
    plot_comparison_memory(comparison, str(output_path / "comparison_memory.png"))
    plot_comparison_radar(comparison, str(output_path / "comparison_radar.png"))

    # Generate markdown report
    report_path = output_path / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write(f"# {comparison['name']}\n\n")
        f.write(f"**Date:** {comparison['timestamp']}\n\n")
        f.write(f"**Backends tested:** {comparison['num_backends']}\n")
        f.write(f"**Videos:** {comparison['num_videos']}\n")
        f.write(f"**Frames per video:** {comparison['frames_per_video']}\n\n")

        f.write("## Rankings\n\n")
        f.write(f"**Fastest (latency):** {' > '.join(comparison['latency_ranking'])}\n\n")
        f.write(f"**Highest throughput:** {' > '.join(comparison['throughput_ranking'])}\n\n")
        f.write(f"**Lowest memory:** {' > '.join(comparison['memory_ranking'])}\n\n")

        f.write("## Results Table\n\n")
        f.write("| Backend | Latency (ms) | Videos/sec | Tokens/sec | Memory (MB) |\n")
        f.write("|---------|--------------|------------|------------|-------------|\n")

        for b in comparison["backend_results"]:
            name = b.get("backend_info", {}).get("display_name", b.get("backend_type", "unknown"))
            f.write(
                f"| {name} | {b['latency_mean_ms']:.1f} | "
                f"{b['videos_per_second']:.4f} | "
                f"{b['tokens_per_second']:.1f} | "
                f"{b['peak_memory_mb']:.0f} |\n"
            )

        f.write("\n## Visualizations\n\n")
        f.write("![Latency Comparison](comparison_latency.png)\n\n")
        f.write("![Throughput Comparison](comparison_throughput.png)\n\n")
        f.write("![Memory Comparison](comparison_memory.png)\n\n")
        f.write("![Radar Chart](comparison_radar.png)\n")

    print(f"Comparison report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results_file", type=str, help="Path to JSON results file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reports",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["latency", "memory", "throughput", "frames", "all"],
        default="all",
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison report (for comparison_*.json files)",
    )
    args = parser.parse_args()

    results = load_results(args.results_file)

    if args.compare or "backend_results" in results:
        # This is a comparison results file
        generate_comparison_report(results, args.output_dir)
    elif args.plot == "all":
        generate_report(results, args.output_dir)
    elif args.plot == "latency":
        plot_latency_distribution(results)
    elif args.plot == "memory":
        plot_memory_timeline(results)
    elif args.plot == "throughput":
        plot_throughput_metrics(results)
    elif args.plot == "frames":
        plot_latency_by_frames(results)


if __name__ == "__main__":
    main()
