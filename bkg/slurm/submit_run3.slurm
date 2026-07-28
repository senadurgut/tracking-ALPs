#!/bin/bash
# Single-job cut flow for the Run 3 parking K_L -> gamma gamma background.
#
# SUBMIT (from this directory):
#   sbatch submit_run3.slurm
#
#SBATCH --job-name=kl_run3
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=00:15:00
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
"$PYTHON" analysis/run3_parking.py
echo "done"
