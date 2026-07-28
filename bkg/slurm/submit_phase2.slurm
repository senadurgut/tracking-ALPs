#!/bin/bash
# Single-job cut flow for the Phase-2 scouting K_L -> gamma gamma background.
# One vectorized pass over kl_sample.csv (analytic gg decay + scouting cuts) -> not an array.
#
# SUBMIT (from this directory):
#   sbatch submit_phase2.slurm
#
#SBATCH --job-name=kl_phase2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
## #SBATCH --partition=work

set -euo pipefail

WORKDIR=/home/export/sdurgut/scratch/alps/tracking_ALPs/bkg
PYTHON=/home/export/sdurgut/micromamba/envs/foresee310/bin/python
cd "$WORKDIR"
mkdir -p logs

echo "host=$(hostname)"
"$PYTHON" analysis/phase2_scouting.py
echo "done"
