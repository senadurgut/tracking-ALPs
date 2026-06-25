#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM array job: one task per ALP mass.
#
# Usage (from the project root /home/export/sdurgut/scratch/alps/tracking_ALPs):
#
#     # 1) Submit one task per mass.  Adjust --array to match len(ma_list)-1
#     #    in your config (config1.json currently has 31 masses → 0..30).
#     sbatch scripts/slurm/run_array.sbatch
#
#     # 2) After all tasks complete, merge the per-mass parts into the
#     #    single CSV that the plotting notebook expects:
#     python scripts/analysis_helpers/merge_parts.py --config configs/config1.json
#
# To override the config without editing this file:
#     sbatch --export=ALL,CONFIG=configs/config2.json scripts/slurm/run_array.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=alp_vbf
#SBATCH --array=0-30                 # adjust to (len(ma_list) - 1)
#SBATCH --time=00:45:00              # per-task wall clock (one mass)
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=logs/analysis_configurable_v2/%A_%a.out
#SBATCH --error=logs/analysis_configurable_v2/%A_%a.err

set -euo pipefail

# Where this script assumes the project lives.  Override by exporting
# PROJECT_ROOT in your environment before `sbatch`.
PROJECT_ROOT="${PROJECT_ROOT:-/home/export/sdurgut/scratch/alps/tracking_ALPs}"
CONFIG="${CONFIG:-configs/config1.json}"
NUM_EVENTS="${NUM_EVENTS:-100000}"
DATA_DIR="${DATA_DIR:-data/cmsrun3-csvs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SEED="${SEED:-1234}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${RESULTS_DIR}"

# Resolve the output dir once for the whole array (race-safe), share via file.
STEM=$(basename "${CONFIG}"); STEM=${STEM%.*}
RUNF="${RESULTS_DIR}/.rundir_${SLURM_ARRAY_JOB_ID}"
if mkdir "${RESULTS_DIR}/.lock_${SLURM_ARRAY_JOB_ID}" 2>/dev/null; then
    python3 -c "import sys;sys.path.insert(0,'analysis');from analyze_configurable_v2 import _run_dir;print(_run_dir('${RESULTS_DIR}','${STEM}'))" > "${RUNF}"
fi
until [ -s "${RUNF}" ]; do sleep 1; done
RUN_DIR=$(cat "${RUNF}")

echo "[task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}] starting on $(hostname) at $(date)"
echo "  script:      analyze_configurable_parallel.py"
echo "  config:      ${CONFIG}"
echo "  num-events:  ${NUM_EVENTS}"
echo "  data-dir:    ${DATA_DIR}"
echo "  run-dir:     ${RUN_DIR}"
echo "  seed:        ${SEED}"
echo "  era:         $(jq -r '.era' "${CONFIG}")"
echo "  analysis-mode: $(jq -r '.analysis_mode' "${CONFIG}")"

python3 analysis/analyze_configurable_parallel.py \
    --config       "${CONFIG}" \
    --mass-index   "${SLURM_ARRAY_TASK_ID}" \
    --num-events   "${NUM_EVENTS}" \
    --data-dir     "${DATA_DIR}" \
    --run-dir      "${RUN_DIR}" \
    --seed         "${SEED}"

echo "[task ${SLURM_ARRAY_TASK_ID}] done at $(date)"
