#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python make_synthetic.py --output data/synthetic
python run_comparison.py --data data/synthetic --config configs/smoke.yaml \
  --device cpu --workdir runs/synthetic_smoke --smoke-steps 2

