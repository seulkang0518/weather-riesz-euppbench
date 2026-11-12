#!/usr/bin/env bash
# Resource names follow the user's existing UCL H100 submission scripts.
# Confirm this allocation provides four GPUs on ONE node (torchrun is single-node).
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N eupp_weather
#$ -P aihub_ucl
#$ -pe gpu 4
#$ -l tmem=32G,h_rt=48:00:00,gpu=true,gpu_type=h100
set -euo pipefail
: "${WEATHER_PROJECT:?Set WEATHER_PROJECT to this new project on the cluster}"
: "${WEATHER_DATA:?Set WEATHER_DATA to the prepared data directory}"
: "${WEATHER_RUN:?Set WEATHER_RUN to a new results directory}"
cd "$WEATHER_PROJECT"
# Activate your tested Python environment before qsub (or add its activation here).
PYTHON="${WEATHER_PYTHON:-python}"
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Allocate exactly four GPUs using the site-approved request'
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(4)])
PY
"$PYTHON" run_comparison.py --data "$WEATHER_DATA" --config configs/temperature48.yaml \
  --gpus 4 --device cuda --workdir "$WEATHER_RUN"
