#!/bin/bash
# run_benchmark.sh
# Run VLM benchmark for Qwen3-4B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run benchmark
echo "Starting VLM Benchmark..."
python src/benchmark.py \
    --config configs/benchmark_config.yaml \
    --video-dir data/motion \
    --output-dir results

# Generate visualizations from latest results
LATEST_RESULT=$(ls -t results/benchmark_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_RESULT" ]; then
    echo "Generating visualizations..."
    python src/visualize.py "$LATEST_RESULT" --output-dir results/reports
fi

echo "Benchmark complete! Results saved to results/"
