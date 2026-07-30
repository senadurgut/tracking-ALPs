#!/bin/bash
# Signal-vs-K_L plots (pT, displacement, selection-text panel). One array task per config.
#
# SUBMIT (from bkg/):
#   sbatch slurm/submit_sig_vs_bkg.slurm                       # default: config1..config8 (fraction/bin)
#
#   # exact full-K_L yields (events @ lumi / bin):
#   sbatch --export=ALL,MAXKL=0,NORM=events slurm/submit_sig_vs_bkg.slurm
#
#   # any set of N configs -- set --array to match the count:
#   sbatch --array=0-2 --export=ALL,CONFIGS="configs/a.json configs/b.json configs/c.json" slurm/submit_sig_vs_bkg.slurm
#
# Env knobs: CONFIGS (space-separated list), MAXKL (0=all rows), NORM (fraction|events), NOACT (1=force activity off).
#
#SBATCH --job-name=sig_vs_bkg
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-8
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
## #SBATCH --partition=work

set -euo pipefail

WORKDIR=/home/export/sdurgut/scratch/alps/tracking_ALPs/bkg
PYTHON=/home/export/sdurgut/micromamba/envs/foresee310/bin/python
cd "$WORKDIR"
mkdir -p logs plots

CONFIGS_STR=${CONFIGS:-"configs/config1.json configs/config2.json configs/config3.json configs/config4.json configs/config5.json configs/config6.json configs/config7.json configs/config8.json configs/config9.json"}
read -ra CFGS <<< "$CONFIGS_STR"
CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}
MAXKL=${MAXKL:-400000}
NORM=${NORM:-fraction}

EXTRA=()
[ "${NOACT:-0}" != "0" ] && EXTRA+=(--no-activity-cut)

echo "host=$(hostname)  config=$CFG  max_kl=$MAXKL  norm=$NORM"
"$PYTHON" plotting/plot_sig_vs_bkg.py --config "$CFG" --max-kl "$MAXKL" --norm "$NORM" ${EXTRA[@]+"${EXTRA[@]}"}
echo "done -> plots/$(basename "$CFG" .json).png"
