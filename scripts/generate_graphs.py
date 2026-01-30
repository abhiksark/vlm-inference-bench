# scripts/generate_graphs.py
"""Generate benchmark visualization graphs for README."""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data by precision category
data_fp8 = {
    'vLLM FP8': {'throughput': 52.7, 'latency': 2819, 'memory': 38.3},
    'SGLang FP8': {'throughput': 34.5, 'latency': 5017, 'memory': 43.1},
}

data_bf16 = {
    'vLLM BF16': {'throughput': 38.6, 'latency': 3658, 'memory': 38.2},
    'SGLang BF16': {'throughput': 29.8, 'latency': 5458, 'memory': 43.1},
    'Ollama F16': {'throughput': 6.6, 'latency': 4423, 'memory': 12.1},
}

data_8bit = {
    'Ollama Q8': {'throughput': 8.0, 'latency': 4081, 'memory': 8.4},
}

data_4bit = {
    'Ollama Q4': {'throughput': 52.6, 'latency': 4891, 'memory': 13.8},
}

# All data combined
data_all = {
    'vLLM FP8': {'throughput': 52.7, 'latency': 2819, 'memory': 38.3, 'precision': 'FP8'},
    'Ollama Q4': {'throughput': 52.6, 'latency': 4891, 'memory': 13.8, 'precision': '4-bit'},
    'vLLM BF16': {'throughput': 38.6, 'latency': 3658, 'memory': 38.2, 'precision': 'BF16'},
    'SGLang FP8': {'throughput': 34.5, 'latency': 5017, 'memory': 43.1, 'precision': 'FP8'},
    'SGLang BF16': {'throughput': 29.8, 'latency': 5458, 'memory': 43.1, 'precision': 'BF16'},
    'Ollama Q8': {'throughput': 8.0, 'latency': 4081, 'memory': 8.4, 'precision': '8-bit'},
    'Ollama F16': {'throughput': 6.6, 'latency': 4423, 'memory': 12.1, 'precision': 'BF16'},
}

# Color scheme by backend
colors = {
    'vLLM FP8': '#2ecc71',
    'vLLM BF16': '#27ae60',
    'SGLang FP8': '#3498db',
    'SGLang BF16': '#2980b9',
    'Ollama Q4': '#e74c3c',
    'Ollama Q8': '#c0392b',
    'Ollama F16': '#e67e22',
}

plt.style.use('seaborn-v0_8-whitegrid')


def create_precision_chart(data, title, filename, figsize=(8, 4)):
    """Create a combined throughput + latency chart for a precision category."""
    backends = list(data.keys())
    throughputs = [data[b]['throughput'] for b in backends]
    latencies = [data[b]['latency'] / 1000 for b in backends]
    memories = [data[b]['memory'] for b in backends]
    bar_colors = [colors[b] for b in backends]

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]), dpi=150)

    # Throughput chart
    ax1 = axes[0]
    bars1 = ax1.barh(backends, throughputs, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars1, throughputs):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                 va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Tokens/s', fontsize=11)
    ax1.set_title('Throughput (↑ better)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, max(throughputs) * 1.2)
    ax1.invert_yaxis()

    # Latency chart
    ax2 = axes[1]
    bars2 = ax2.barh(backends, latencies, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars2, latencies):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s',
                 va='center', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Seconds', fontsize=11)
    ax2.set_title('Latency (↓ better)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(latencies) * 1.2)
    ax2.invert_yaxis()
    ax2.set_yticklabels([])

    # Memory chart
    ax3 = axes[2]
    bars3 = ax3.barh(backends, memories, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars3, memories):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}GB',
                 va='center', fontsize=10, fontweight='bold')
    ax3.set_xlabel('GB', fontsize=11)
    ax3.set_title('Memory (↓ better)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, max(memories) * 1.2)
    ax3.invert_yaxis()
    ax3.set_yticklabels([])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'assets/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: assets/{filename}')


def create_overall_comparison():
    """Create overall comparison chart sorted by throughput."""
    # Sort by throughput
    sorted_backends = sorted(data_all.keys(), key=lambda x: data_all[x]['throughput'], reverse=True)
    throughputs = [data_all[b]['throughput'] for b in sorted_backends]
    latencies = [data_all[b]['latency'] / 1000 for b in sorted_backends]
    memories = [data_all[b]['memory'] for b in sorted_backends]
    bar_colors = [colors[b] for b in sorted_backends]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    # Throughput
    ax1 = axes[0]
    bars1 = ax1.barh(sorted_backends, throughputs, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars1, throughputs):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                 va='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Tokens/s', fontsize=11)
    ax1.set_title('Throughput (↑ better)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, max(throughputs) * 1.15)
    ax1.invert_yaxis()

    # Latency
    ax2 = axes[1]
    bars2 = ax2.barh(sorted_backends, latencies, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars2, latencies):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s',
                 va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Seconds', fontsize=11)
    ax2.set_title('Latency (↓ better)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(latencies) * 1.15)
    ax2.invert_yaxis()
    ax2.set_yticklabels([])

    # Memory
    ax3 = axes[2]
    bars3 = ax3.barh(sorted_backends, memories, color=bar_colors, edgecolor='white')
    for bar, val in zip(bars3, memories):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}GB',
                 va='center', fontsize=9, fontweight='bold')
    ax3.set_xlabel('GB', fontsize=11)
    ax3.set_title('Memory (↓ better)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, max(memories) * 1.15)
    ax3.invert_yaxis()
    ax3.set_yticklabels([])

    fig.suptitle('Overall Comparison (sorted by throughput)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/overall_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/overall_comparison.png')


def create_efficiency_chart():
    """Create efficiency scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for backend, d in data_all.items():
        ax.scatter(d['memory'], d['throughput'], s=200, c=colors[backend],
                   label=backend, edgecolors='white', linewidth=1.5, zorder=3)

    # Add labels
    for backend, d in data_all.items():
        offset_x = 1 if d['memory'] < 30 else -8
        offset_y = 1.5 if d['throughput'] > 20 else -2
        ax.annotate(backend, (d['memory'], d['throughput']),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_ylabel('Tokens per Second', fontsize=12)
    ax.set_title('Efficiency: Throughput vs Memory', fontsize=14, fontweight='bold')

    # Quadrant annotations
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.3)
    ax.text(10, 55, 'Best\n(High throughput\nLow memory)', fontsize=9, alpha=0.6, ha='center')
    ax.text(40, 5, 'Worst\n(Low throughput\nHigh memory)', fontsize=9, alpha=0.6, ha='center')

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 60)
    ax.legend(loc='center right', fontsize=9)

    plt.tight_layout()
    plt.savefig('assets/efficiency_scatter.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/efficiency_scatter.png')


if __name__ == '__main__':
    # Per-precision charts
    create_precision_chart(data_fp8, 'FP8 Precision: vLLM vs SGLang', 'fp8_comparison.png')
    create_precision_chart(data_bf16, 'BF16 Precision: vLLM vs SGLang vs Ollama', 'bf16_comparison.png')
    create_precision_chart(data_8bit, '8-bit Precision: Ollama Q8', '8bit_comparison.png', figsize=(6, 2))
    create_precision_chart(data_4bit, '4-bit Precision: Ollama Q4', '4bit_comparison.png', figsize=(6, 2))

    # Overall charts
    create_overall_comparison()
    create_efficiency_chart()

    print('All graphs generated!')
