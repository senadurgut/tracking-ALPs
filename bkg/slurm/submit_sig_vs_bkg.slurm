#!/bin/bash
# Signal-vs-K_L distribution plots (mass, pT, displacement) with the full config7/config10
# selection + activity<=0. Array of 2: task 0 = run3, task 1 = phase2 (independent, run in parallel).
#
# SUBMIT (from this directory):
#   sbatch submit_sig_vs_bkg.slurm
#   # optional: subsample fewer/more K_L rows (0 = all):
#   sbatch --export=ALL,MAXKL=1000000 submit_sig_vs_bkg.slurm
#
#SBATCH --job-name=sig_vs_bkg
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
## #SBATCH --partition=work

set -euo pipefail

WORKDIR=/home/export/sdurgut/scratch/alps/tracking_ALPs/bkg
PYTHON=/home/export/sdurgut/micromamba/envs/foresee310/bin/python
cd "$WORKDIR"
mkdir -p logs plots

ANALYSES=(run3 phase2)
A=${ANALYSES[$SLURM_ARRAY_TASK_ID]}
MAXKL=${MAXKL:-400000}

echo "host=$(hostname)  analysis=$A  max_kl=$MAXKL"
"$PYTHON" plot_sig_vs_bkg.py --analysis "$A" --max-kl "$MAXKL"
echo "done -> plots/sig_vs_bkg_${A}.png"
