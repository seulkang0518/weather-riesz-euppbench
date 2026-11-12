#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_comparison.py --data "${WEATHER_DATA:-data/temperature48}" \
  --config configs/temperature48.yaml --gpus 1 --device cuda \
  --workdir "${WEATHER_RUN:-runs/temperature48_scafell_seed42}"

