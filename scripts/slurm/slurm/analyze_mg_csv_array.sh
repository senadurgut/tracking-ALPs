#!/usr/bin/env bash
#SBATCH --job-name=analyze_mg_csv
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --qos=cpu
#SBATCH --time=48:00:00
#SBATCH --array=0-27
#SBATCH --output=logs/Analyze_madgraph_output/%x_%A_%a.out
#SBATCH --error=logs/Analyze_madgraph_output/%x_%A_%a.err

# Run Analyze_madgraph_output.py for one mass index per array task.
#
# Prerequisites
# -------------
# - Mass index must match ``ma_list`` in Analyze_madgraph_output.py (0-based).
# - Merged CSVs must exist as merged_ma_<token>GeV.csv under DATA_DIR (same
#   naming as merge_cmsrun3_csvs_by_mass.sh).
#
# Setup
# -----
#   mkdir -p logs
#   # Optional: cap concurrent tasks if many array jobs append to the same
#   # results CSVs (e.g. --array=0-27%8).
#   sbatch slurm/analyze_mg_csv_array.sh
#
# Override paths (optional):
#   export CMSRUN3_CSVS_MERGED_DIR=/path/to/merged/csvs
#   export N_MASS=28   # must equal len(ma_list) and #SBATCH --array upper bound + 1

set -euo pipefail

TRACKING_DIR="/home/export/sdurgut/scratch/alps/tracking-ALPs"
DATA_DIR="${CMSRUN3_CSVS_MERGED_DIR:-${TRACKING_DIR}/data/cmsrun3-csvs-merged}"

# Number of mass indices = length of ma_list in Analyze_madgraph_output.py
N_MASS="${N_MASS:-28}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set (submit with sbatch, not bash)" >&2
  exit 2
fi
if [[ "$SLURM_ARRAY_TASK_ID" -lt 0 || "$SLURM_ARRAY_TASK_ID" -ge "$N_MASS" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((N_MASS - 1))]" >&2
  exit 2
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: merged CSV directory not found: $DATA_DIR" >&2
  exit 2
fi

cd "$TRACKING_DIR"
mkdir -p logs/Analyze_madgraph_output results

export CMSRUN3_CSVS_MERGED_DIR="$DATA_DIR"
# Always write to tracking-ALPs/results (ignore inherited env vars).
export TRACKING_ALP_RESULTS_DIR="$TRACKING_DIR/results"
export PYTHONUNBUFFERED=1

idx="$SLURM_ARRAY_TASK_ID"
echo "[$(date)] SLURM job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} mass_index=${idx}"
echo "DATA_DIR=${CMSRUN3_CSVS_MERGED_DIR}"
echo "TRACKING_ALP_RESULTS_DIR=${TRACKING_ALP_RESULTS_DIR}"

results_total="${TRACKING_ALP_RESULTS_DIR}/results_vbf_total.csv"

# Skip mass points that already have a row in results_vbf_total.csv.
# This lets you re-submit the full array safely; only missing masses do work.
ma_name="$(python3 - "$idx" <<'PY'
import sys

idx = int(sys.argv[1])

# Must match ma_list in Analyze_madgraph_output.py
ma_list = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.1,
    0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1.0, 2.0,
    3.0, 4.0, 5.0, 6.0, 7.0,
    8.0, 9.0, 10.0,
]

ma = ma_list[idx]
token = f"{ma:.4f}".replace(".", "p")
print(f"merged_ma_{token}GeV")
PY
)"

if [[ -f "$results_total" ]]; then
  if grep -q "^${ma_name}," "$results_total"; then
    echo "Already have results row for ${ma_name} in ${results_total}; skipping."
    exit 0
  fi
fi

python3 -u Analyze_madgraph_output.py "$idx"

echo "[$(date)] DONE mass_index=${idx}"
