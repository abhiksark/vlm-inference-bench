# scripts/generate_graphs.py
"""Generate benchmark visualization graphs for README."""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data by precision category
data_fp8 = {
    'vLLM': {'throughput': 52.7, 'latency': 2.82, 'memory': 38.3},
    'SGLang': {'throughput': 34.5, 'latency': 5.02, 'memory': 43.1},
}

data_bf16 = {
    'vLLM': {'throughput': 38.6, 'latency': 3.66, 'memory': 38.2},
    'SGLang': {'throughput': 29.8, 'latency': 5.46, 'memory': 43.1},
    'Ollama F16': {'throughput': 6.6, 'latency': 4.42, 'memory': 12.1},
}

# All data combined for overall chart
data_all = [
    ('vLLM FP8', 52.7, 2.82, 38.3, '#2ecc71'),
    ('Ollama Q4', 52.6, 4.89, 13.8, '#e74c3c'),
    ('vLLM BF16', 38.6, 3.66, 38.2, '#27ae60'),
    ('SGLang FP8', 34.5, 5.02, 43.1, '#3498db'),
    ('SGLang BF16', 29.8, 5.46, 43.1, '#2980b9'),
    ('Ollama Q8', 8.0, 4.08, 8.4, '#c0392b'),
    ('Ollama F16', 6.6, 4.42, 12.1, '#e67e22'),
]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11


def create_comparison_chart(data, title, filename, winner_label=None):
    """Create a comparison chart for a precision category."""
    backends = list(data.keys())
    throughputs = [data[b]['throughput'] for b in backends]
    latencies = [data[b]['latency'] for b in backends]
    memories = [data[b]['memory'] for b in backends]

    # Colors: green for vLLM (winner), blue for SGLang, orange for Ollama
    colors = []
    for b in backends:
        if 'vLLM' in b:
            colors.append('#2ecc71')
        elif 'SGLang' in b:
            colors.append('#3498db')
        else:
            colors.append('#e67e22')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150)

    # Sort by throughput for display
    sorted_idx = np.argsort(throughputs)[::-1]
    sorted_backends = [backends[i] for i in sorted_idx]
    sorted_throughputs = [throughputs[i] for i in sorted_idx]
    sorted_latencies = [latencies[i] for i in sorted_idx]
    sorted_memories = [memories[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    # Throughput chart
    ax1 = axes[0]
    bars1 = ax1.barh(sorted_backends, sorted_throughputs, color=sorted_colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars1, sorted_throughputs):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                 va='center', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Tokens/s')
    ax1.set_title('Throughput (↑ better)', fontweight='bold')
    ax1.set_xlim(0, max(sorted_throughputs) * 1.25)
    ax1.invert_yaxis()

    # Latency chart
    ax2 = axes[1]
    bars2 = ax2.barh(sorted_backends, sorted_latencies, color=sorted_colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars2, sorted_latencies):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s',
                 va='center', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Seconds')
    ax2.set_title('Latency (↓ better)', fontweight='bold')
    ax2.set_xlim(0, max(sorted_latencies) * 1.25)
    ax2.invert_yaxis()

    # Memory chart
    ax3 = axes[2]
    bars3 = ax3.barh(sorted_backends, sorted_memories, color=sorted_colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars3, sorted_memories):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}GB',
                 va='center', fontsize=11, fontweight='bold')
    ax3.set_xlabel('GB')
    ax3.set_title('Memory (↓ better)', fontweight='bold')
    ax3.set_xlim(0, max(sorted_memories) * 1.25)
    ax3.invert_yaxis()

    if winner_label:
        fig.suptitle(f'{title}\n{winner_label}', fontsize=13, fontweight='bold', y=1.05)
    else:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(f'assets/{filename}', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: assets/{filename}')


def create_overall_comparison():
    """Create overall comparison chart sorted by throughput."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150)

    backends = [d[0] for d in data_all]
    throughputs = [d[1] for d in data_all]
    latencies = [d[2] for d in data_all]
    memories = [d[3] for d in data_all]
    colors = [d[4] for d in data_all]

    # Throughput
    ax1 = axes[0]
    bars1 = ax1.barh(backends, throughputs, color=colors, edgecolor='white', height=0.7)
    for bar, val in zip(bars1, throughputs):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                 va='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Tokens/s')
    ax1.set_title('Throughput (↑ better)', fontweight='bold')
    ax1.set_xlim(0, max(throughputs) * 1.2)
    ax1.invert_yaxis()

    # Latency
    ax2 = axes[1]
    bars2 = ax2.barh(backends, latencies, color=colors, edgecolor='white', height=0.7)
    for bar, val in zip(bars2, latencies):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}s',
                 va='center', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Seconds')
    ax2.set_title('Latency (↓ better)', fontweight='bold')
    ax2.set_xlim(0, max(latencies) * 1.2)
    ax2.invert_yaxis()

    # Memory
    ax3 = axes[2]
    bars3 = ax3.barh(backends, memories, color=colors, edgecolor='white', height=0.7)
    for bar, val in zip(bars3, memories):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}GB',
                 va='center', fontsize=10, fontweight='bold')
    ax3.set_xlabel('GB')
    ax3.set_title('Memory (↓ better)', fontweight='bold')
    ax3.set_xlim(0, max(memories) * 1.2)
    ax3.invert_yaxis()

    fig.suptitle('Overall Comparison (sorted by throughput)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/overall_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/overall_comparison.png')


def create_efficiency_chart():
    """Create efficiency scatter plot with dot size proportional to precision."""
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

    # Precision to dot size mapping (more bits = larger dot)
    precision_sizes = {
        '4-bit': 150,
        '8-bit': 250,
        'FP8': 350,
        'BF16': 450,
    }

    # Data with precision info: (name, throughput, latency, memory, color, precision)
    data_with_precision = [
        ('vLLM FP8', 52.7, 2.82, 38.3, '#2ecc71', 'FP8'),
        ('Ollama Q4', 52.6, 4.89, 13.8, '#e74c3c', '4-bit'),
        ('vLLM BF16', 38.6, 3.66, 38.2, '#27ae60', 'BF16'),
        ('SGLang FP8', 34.5, 5.02, 43.1, '#3498db', 'FP8'),
        ('SGLang BF16', 29.8, 5.46, 43.1, '#2980b9', 'BF16'),
        ('Ollama Q8', 8.0, 4.08, 8.4, '#c0392b', '8-bit'),
        ('Ollama F16', 6.6, 4.42, 12.1, '#e67e22', 'BF16'),
    ]

    # Plot each point
    for name, throughput, latency, memory, color, precision in data_with_precision:
        size = precision_sizes[precision]
        ax.scatter(memory, throughput, s=size, c=color, edgecolors='white',
                   linewidth=2, zorder=3, alpha=0.85)

        # Position labels to avoid overlap
        if name == 'Ollama Q4':
            ax.annotate(name, (memory, throughput), xytext=(-65, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        elif name == 'vLLM FP8':
            ax.annotate(name, (memory, throughput), xytext=(12, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        elif name == 'vLLM BF16':
            ax.annotate(name, (memory, throughput), xytext=(12, -5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        elif name == 'SGLang FP8':
            ax.annotate(name, (memory, throughput), xytext=(12, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        elif name == 'SGLang BF16':
            ax.annotate(name, (memory, throughput), xytext=(12, -12), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        elif name == 'Ollama Q8':
            ax.annotate(name, (memory, throughput), xytext=(12, 3), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        elif name == 'Ollama F16':
            ax.annotate(name, (memory, throughput), xytext=(12, -8), textcoords='offset points',
                        fontsize=10, fontweight='bold')

    ax.set_xlabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_ylabel('Tokens per Second', fontsize=12)
    ax.set_title('Efficiency: Throughput vs Memory\n(dot size = precision bits)', fontsize=14, fontweight='bold')

    # Quadrant lines
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(x=25, color='gray', linestyle='--', alpha=0.4)

    # Quadrant labels
    ax.text(8, 60, 'BEST\nHigh speed, Low memory', fontsize=10, ha='center',
            color='#27ae60', fontweight='bold', alpha=0.8)
    ax.text(35, 15, 'WORST\nLow speed\nHigh memory', fontsize=10, ha='center',
            color='#e74c3c', fontweight='bold', alpha=0.6)

    # Add precision size legend
    legend_elements = [
        plt.scatter([], [], s=precision_sizes['BF16'], c='gray', alpha=0.6, label='BF16 (16-bit)'),
        plt.scatter([], [], s=precision_sizes['FP8'], c='gray', alpha=0.6, label='FP8 (8-bit)'),
        plt.scatter([], [], s=precision_sizes['8-bit'], c='gray', alpha=0.6, label='Q8 (8-bit)'),
        plt.scatter([], [], s=precision_sizes['4-bit'], c='gray', alpha=0.6, label='Q4 (4-bit)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', title='Dot Size = Precision',
              fontsize=9, title_fontsize=10, framealpha=0.95)

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 65)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/efficiency_scatter.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/efficiency_scatter.png')


if __name__ == '__main__':
    # Per-precision comparison charts (only where there's actual competition)
    create_comparison_chart(data_fp8, 'FP8 Precision', 'fp8_comparison.png',
                           'vLLM wins: 53% faster throughput, 44% lower latency')
    create_comparison_chart(data_bf16, 'BF16 Precision', 'bf16_comparison.png',
                           'vLLM wins: 30% faster than SGLang, 6x faster than Ollama')

    # Overall charts
    create_overall_comparison()
    create_efficiency_chart()

    print('All graphs generated!')
