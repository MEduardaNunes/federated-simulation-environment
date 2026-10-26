#!/usr/bin/env bash
#SBATCH --job-name=flower-sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/mariaeduarda.costa/slurm/flower-sim%j.out
#SBATCH --error=/home/users/mariaeduarda.costa/slurm/flower-sim%j.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mecosta961@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate sim-env

export HF_DATASETS_OFFLINE=1
export RAY_DISABLE_DASHBOARD=1

make run-complete