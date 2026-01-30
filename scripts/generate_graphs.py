# scripts/generate_graphs.py
"""Generate benchmark visualization graphs for README."""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data from results
data = {
    'vLLM FP8': {'throughput': 52.7, 'latency': 2819, 'memory': 38.3, 'precision': 'FP8'},
    'Ollama Q4': {'throughput': 52.6, 'latency': 4891, 'memory': 13.8, 'precision': '4-bit'},
    'vLLM BF16': {'throughput': 38.6, 'latency': 3658, 'memory': 38.2, 'precision': 'BF16'},
    'SGLang FP8': {'throughput': 34.5, 'latency': 5017, 'memory': 43.1, 'precision': 'FP8'},
    'SGLang BF16': {'throughput': 29.8, 'latency': 5458, 'memory': 43.1, 'precision': 'BF16'},
    'Ollama Q8': {'throughput': 8.0, 'latency': 4081, 'memory': 8.4, 'precision': '8-bit'},
    'Ollama F16': {'throughput': 6.6, 'latency': 4423, 'memory': 12.1, 'precision': 'BF16'},
}

backends = list(data.keys())
throughputs = [data[b]['throughput'] for b in backends]
latencies = [data[b]['latency'] / 1000 for b in backends]  # Convert to seconds
memories = [data[b]['memory'] for b in backends]

# Color scheme
colors = {
    'vLLM FP8': '#2ecc71',
    'vLLM BF16': '#27ae60',
    'SGLang FP8': '#3498db',
    'SGLang BF16': '#2980b9',
    'Ollama Q4': '#e74c3c',
    'Ollama Q8': '#c0392b',
    'Ollama F16': '#e67e22',
}
bar_colors = [colors[b] for b in backends]

plt.style.use('seaborn-v0_8-whitegrid')
fig_params = {'figsize': (10, 6), 'dpi': 150}


def save_throughput_chart():
    """Throughput comparison bar chart."""
    fig, ax = plt.subplots(**fig_params)

    # Sort by throughput
    sorted_idx = np.argsort(throughputs)[::-1]
    sorted_backends = [backends[i] for i in sorted_idx]
    sorted_throughputs = [throughputs[i] for i in sorted_idx]
    sorted_colors = [bar_colors[i] for i in sorted_idx]

    bars = ax.barh(sorted_backends, sorted_throughputs, color=sorted_colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, sorted_throughputs):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Tokens per Second', fontsize=12)
    ax.set_title('Throughput Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(sorted_throughputs) * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('assets/throughput_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/throughput_comparison.png')


def save_latency_chart():
    """Latency comparison bar chart."""
    fig, ax = plt.subplots(**fig_params)

    # Sort by latency (ascending - lower is better)
    sorted_idx = np.argsort(latencies)
    sorted_backends = [backends[i] for i in sorted_idx]
    sorted_latencies = [latencies[i] for i in sorted_idx]
    sorted_colors = [bar_colors[i] for i in sorted_idx]

    bars = ax.barh(sorted_backends, sorted_latencies, color=sorted_colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, sorted_latencies):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}s',
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Latency P50 (seconds)', fontsize=12)
    ax.set_title('Latency Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(sorted_latencies) * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('assets/latency_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/latency_comparison.png')


def save_memory_chart():
    """Memory usage comparison bar chart."""
    fig, ax = plt.subplots(**fig_params)

    # Sort by memory (ascending - lower is better)
    sorted_idx = np.argsort(memories)
    sorted_backends = [backends[i] for i in sorted_idx]
    sorted_memories = [memories[i] for i in sorted_idx]
    sorted_colors = [bar_colors[i] for i in sorted_idx]

    bars = ax.barh(sorted_backends, sorted_memories, color=sorted_colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, sorted_memories):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f} GB',
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_title('Memory Usage (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(sorted_memories) * 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('assets/memory_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/memory_comparison.png')


def save_efficiency_chart():
    """Throughput vs Memory scatter plot."""
    fig, ax = plt.subplots(**fig_params)

    for backend in backends:
        d = data[backend]
        ax.scatter(d['memory'], d['throughput'], s=200, c=colors[backend],
                   label=backend, edgecolors='white', linewidth=1.5, zorder=3)

    # Add labels
    for backend in backends:
        d = data[backend]
        offset_x = 1 if d['memory'] < 30 else -8
        offset_y = 1.5 if d['throughput'] > 20 else -2
        ax.annotate(backend, (d['memory'], d['throughput']),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_ylabel('Tokens per Second', fontsize=12)
    ax.set_title('Efficiency: Throughput vs Memory', fontsize=14, fontweight='bold')

    # Add quadrant lines
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.3)

    # Annotations for quadrants
    ax.text(10, 55, 'Best\n(High throughput, Low memory)', fontsize=9, alpha=0.6, ha='center')
    ax.text(40, 5, 'Worst\n(Low throughput, High memory)', fontsize=9, alpha=0.6, ha='center')

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 60)
    ax.legend(loc='center right', fontsize=9)

    plt.tight_layout()
    plt.savefig('assets/efficiency_scatter.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/efficiency_scatter.png')


if __name__ == '__main__':
    save_throughput_chart()
    save_latency_chart()
    save_memory_chart()
    save_efficiency_chart()
    print('All graphs generated!')
