#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="$(pwd)/.mplconfig"
python3 -m src.benchmark_runner --task A
python3 -m src.benchmark_runner --task B
python3 -m src.benchmark_runner --task C
python3 -m src.benchmark_runner --task D
python3 -m src.benchmark_runner --task E
python3 -m src.benchmark_runner --task F
