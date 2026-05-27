#!/bin/bash
set -e

echo "Running memory benchmark..."
python benchmarks/bench_memory.py

echo "Running latency benchmark..."
python benchmarks/bench_latency.py

echo "Running throughput benchmark..."
python benchmarks/bench_throughput.py

echo "All benchmarks complete. Results saved to results/"