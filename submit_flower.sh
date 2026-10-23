#!/usr/bin/env bash
#SBATCH --job-name=flower-sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-flower-%j.out
#SBATCH --error=slurm-flower-%j.err
#SBATCH --gres=gpu:1

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="${FLOWER_VENV:-"$PROJECT_DIR/../.venv"}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    printf 'Python virtual environment not found: %s\n' "$VENV_DIR" >&2
    printf 'Create it once and install the project dependencies before submitting.\n' >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

if ! command -v flwr >/dev/null 2>&1; then
    printf 'The Flower CLI is not available in the environment: %s\n' "$VENV_DIR" >&2
    exit 1
fi
python - <<'PY'
import datasets
import flwr
import flwr_datasets
import gymnasium
import matplotlib
import numpy
import pandas
import torch
import torchvision
PY
if ! command -v flock >/dev/null 2>&1; then
    printf 'Required command not found: flock (used to protect per-run result files).\n' >&2
    exit 1
fi

mkdir -p "$PROJECT_DIR/results"
exec 9> "$PROJECT_DIR/results/.flower-slurm.lock"
if ! flock -n 9; then
    printf 'Another Flower job is using this checkout; refusing to risk overwriting its results.\n' >&2
    exit 1
fi

# Keep Flower's managed local SuperLink state and API port specific to this job.
JOB_ID="${SLURM_JOB_ID:-$$}"
export FLWR_HOME="$PROJECT_DIR/results/.flower-home-$JOB_ID"
export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT="$((20000 + JOB_ID % 40000))"
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

# The app selects CUDA whenever it is visible, but this job requests no GPU.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

make run-simple