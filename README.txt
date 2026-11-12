EUPPBENCH: ONE-PASS RIESZ / ENERGY VS FLOW MATCHING
=================================================
Separate project. It does not modify weather-riesz, DMMD, or Hudson's repository.
Read audit/README.txt for the paper/code audit and exact adaptation differences.

WHAT THIS TESTS
---------------
Same station-token Transformer, forecast conditioning, normalized targets and
data splits; four training objectives:
  energy          one-pass generator, conditional energy-score loss
  riesz_softmax   one-pass generator, stopped softmax Riesz field
  riesz_uniform   one-pass generator, uniform-neighbor Riesz control
  flow_matching   velocity regression, iterative generation (default 16 calls)

All models generate corrections to a known forecast mean in anomaly coordinates:
  y_hat = baseline + residual_scale * generated_correction
Endpoint arms predict the correction in one call. The flow arm starts from noise
and integrates a predicted velocity. A scenario is all stations/variables for ONE
forecast initialization and lead, not a joint trajectory through all lead times.
Particle comparisons never pool unrelated forecast cases. No MAE is used.

Pilot: 122 stations x temperature at 48 h; 8 TRAINING particles for endpoint arms.
Full configuration: 122 stations x temperature+gust, 20 lead times; 50 training
particles. It is a more expensive follow-up, not a validated paper reproduction.
All final ensembles contain 51 scenarios PER trained model, independent of the
training-particle count. Main summaries average scores across seeds rather than
pooling multiple trained models. FMAP paper's ten-model ESGM recipe is different.

INSTALL IN A SEPARATE ENVIRONMENT
---------------------------------
Use Python 3.11 and a CUDA-compatible PyTorch >=2.5. Preserve the working dmmd
environment while that experiment is running. For example, on scafell:

    conda create -n eupp-riesz --clone dmmd
    conda activate eupp-riesz
    cd /home/zongchen/weather-riesz-euppbench
    python -m pip install -r requirements.txt
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    python -m pytest -q

Cloning an environment consumes additional disk space. The requirements accept
the already working Torch 2.5.1+cu121; they do not require a driver upgrade.
Xarray is capped below 2025.1 and Zarr below 3 to read this release consistently.
No aqueduct, Hydra, MLflow, external feature encoder, or gridded archive is needed.

FIRST: SMALL CPU WIRING CHECK
-----------------------------
From the new project directory (any machine):

    bash scripts/smoke_cpu.sh

This creates explicitly synthetic 122-station, two-variable data, then performs
two optimizer steps for all four objectives and saves five-member validation
forecasts. These are NOT forecast quality results. Output:

    runs/synthetic_smoke/summary_val.json

The smoke script expects fresh data/run directories. To resume/recheck an existing
smoke run without recreating data:

    python run_comparison.py --data data/synthetic --config configs/smoke.yaml \
      --device cpu --workdir runs/synthetic_smoke --smoke-steps 2

DOWNLOAD AND PREPARE REAL STATION DATA
------------------------------------
Do this on the machine that will hold the dataset. First inspect the plan:

    python download_data.py --root data/raw

The verified archive is 18.3 GB; ZIP + expanded contents need about 36.6 GB
before prepared arrays. Allow roughly 45-50 GB free for the first experiment.
No full-data download occurs until --download is explicitly included:

    python download_data.py --root data/raw --profile temperature48 --download
    python prepare_data.py --raw data/raw --profile temperature48 \
      --output data/temperature48

The downloader supports resumable partial downloads, validates the full published
MD5 and size, and extracts only requested station stores. No full gridded data.
Temperature pilot reads five predictors: t2m,u10,v10,tcc,tcwv; this is a declared
reduced-conditioning experiment, not the paper's complete predictor set.

Expected counts before fully missing-case removal: 3,553 train initializations,
627 validation, 640 test. All have 122 station outputs at 48 h. Preparation checks
station/time/lead alignment, recreates actual reforecast years, keeps partial
observations, fits statistics ONLY on train years, and writes immutable
memory-mapped arrays plus file hashes. Never rewrite a prepared dataset used by
an existing checkpoint. Choose another output directory for changed preparation.
An interrupted preparation directory is deliberately incomplete and refused by
training; preserve it for diagnosis or use a new output directory.

Validation years are 2003,2010,2016. Operational test initialization dates are
2017-04-01 through 2018-12-31. These are not the years from our old 10-station task.
No observation-missingness label is supplied as a model input. Masks only exclude
unobserved target coordinates from training/scoring.

SCAFELL: ONE RTX 4090
---------------------
Run the synthetic GPU check first (new run directory):

    python run_comparison.py --data data/synthetic --config configs/smoke.yaml \
      --device cuda --workdir runs/synthetic_cuda --smoke-steps 2

Then launch the first REAL-data validation comparison:

    nohup bash scripts/scafell.sh > eupp_temperature48.log 2>&1 &
    tail -f eupp_temperature48.log

This uses configs/temperature48.yaml and seed 42, 50 epochs, all four arms.
Global batch is 64 forecast cases using microbatches of two and accumulation.
Width 256 with activation checkpointing is a pilot capacity, not the paper's 1024.
bf16 is enabled only when CUDA supports it; loss/distance calculations use fp32.
GPU memory/timing must be measured on the actual machine; no CUDA run was done
locally. If needed, reduce microbatch_cases in a NEW config/run directory.

Override paths without editing the launcher:

    WEATHER_DATA=/absolute/prepared/path WEATHER_RUN=/absolute/new/run/path \
      bash scripts/scafell.sh

UCL: FOUR H100s ON ONE NODE
---------------------------
The SGE resource names follow your existing aihub_ucl/H100 scripts. The launcher
checks that exactly four GPUs are visible and uses single-node torchrun. Example:

    export WEATHER_PROJECT=/home/zongchen/weather-riesz-euppbench
    export WEATHER_DATA=/SAN/intelsys/imagenet_mmd_flow/euppbench/temperature48
    export WEATHER_RUN=/SAN/intelsys/imagenet_mmd_flow/euppbench/runs/temperature48_4gpu_seed42
    export WEATHER_PYTHON=/home/zongchen/miniconda3/envs/eupp-riesz/bin/python
    cd "$WEATHER_PROJECT"
    qsub -V scripts/ucl_sge.sh

Those /SAN paths are EXAMPLES, not directories discovered or created on UCL.
Use your actual writable storage. This job does not download data on compute nodes.
One-/four-GPU jobs keep batch 64 via accumulation, but Riesz's normalization is
local to each microbatch/rank. Keep hardware partition fixed within each objective
comparison; do not claim bitwise equivalence across GPU counts.

OUTPUTS AND RESUME
------------------
For every objective/seed:
    manifest.json, train.jsonl, latest.pt, best.pt, runner.log
    evaluation_val/metrics.json, predictions_anomaly.npy, per_case_scores.npz

Overall:
    runs/temperature48_scafell_seed42/summary_val.json

Training checkpoints use the same fixed-noise EMPIRICAL validation ES across
objectives. Training ES estimator (unbiased vs empirical) is separately explicit
in YAML. Riesz surrogate values cannot be compared as forecasting scores.
Training logs record gradients, LR and epochs. Do not run concurrent writers into
the same workdir. Rerun the identical launcher to resume epoch checkpoints.
Configs, data identity, world size and training source hashes must match. Changes
require a new workdir, never editing old manifests or bypassing checks.

Scores include empirical ES, full VS, coordinate CRPS, and five-neighbor LVS,
in BOTH anomaly and physical units. Full-profile joint LVS includes cross-variable
pairs. Missing-coordinate/pair masks are explicit. Input normalization has no
validation/test fitting. Raw baseline is uncorrected IFS, unlike the paper's
debiased baseline. Inference timing records device, batch/chunk size and scope;
it excludes I/O, warmup and scoring, so it is not end-to-end operational latency.

TEST EVALUATION IS A SEPARATE, EXPLICIT STEP
------------------------------------------
Inspect validation results and freeze the recipe first. With the IDENTICAL data,
config, methods, seeds and hardware settings used to train:

    python run_comparison.py --data data/temperature48 \
      --config configs/temperature48.yaml --device cuda --gpus 1 \
      --workdir runs/temperature48_scafell_seed42 --evaluate-test

This reuses best checkpoints and generates fixed-size ensembles on the operational
test dates. It does not retrain or select temperatures/LR from test metrics.
To add seeds 43/44, use --seeds 42 43 44 with a NEW workdir; no additional seeds
are silently appended to a completed comparison. Same validation tuning budget
for competing methods is needed before any optimality claim.

FULL JOINT FOLLOW-UP
--------------------
Only after reviewing real pilot data and learning curves:

    python download_data.py --root data/raw --profile full --download
    python prepare_data.py --raw data/raw --profile full --output data/full_joint
    python run_comparison.py --data data/full_joint --config configs/full_joint.yaml \
      --device cuda --gpus 4 --workdir runs/full_joint_4gpu_seed42

The existing ZIP is reverified/reused; additional stores are extracted. This is
much more expensive (all positive leads, both variables, wider network, 50 training
particles). Full-source special cases and target coverage still require a real
full-data audit. An entire station-variable with no training observations is an
explicit error; the code does not silently impute a target distribution.

LIMITS OF THE CURRENT VALIDATION
--------------------------------
See validation/ for numerical tests, synthetic end-to-end checks and the real
public metadata/one-station fragment check. Those checks are not full EUPPBench
training, do not establish forecasting skill, and do not validate 4090/H100
throughput or memory. The official FMAP architecture/recipe has differences
listed in audit/README.txt; call this reference FMAP-inspired until independently
reproduced. Larger dimension does not guarantee Riesz will outperform energy.
