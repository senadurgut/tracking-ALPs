#!/bin/bash
# SLURM job-array worker for the eps_VBF Pythia run.
# Each array task runs ONE Pythia process (MODE=hard or MODE=soft) with its own seed,
# writes JSON counters to $OUTDIR/json, and (soft + DUMP=1) a K_L CSV to $OUTDIR/csv.
#
# SUBMIT (from this directory):
#   mkdir -p data/out/json data/out/csv data/out/vbfcsv logs
#   # ~2e7 hard events for eps_VBF (400 tasks x 50k):
#   sbatch --array=0-399 --export=ALL,MODE=hard,NEV=50000,OUTDIR=data/out submit_eps_vbf.slurm
#   # ~2e6 soft events + K_L dump (40 tasks x 50k):
#   sbatch --array=0-39  --export=ALL,MODE=soft,NEV=50000,OUTDIR=data/out,DUMP=1 submit_eps_vbf.slurm
#
# When BOTH arrays are done:
#   python merge_eps_vbf.py data/out/json
#   ./concat_kl_csv.sh data/out/csv data/kl_sample.csv        # (strips repeated headers)
#
#SBATCH --job-name=pythia_kl
#SBATCH --output=logs/pythia_kl/%x_%A_%a.out
#SBATCH --error=logs/pythia_kl/%x_%A_%a.err
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
## #SBATCH --partition=work

set -euo pipefail

WORKDIR=/home/export/sdurgut/scratch/alps/tracking_ALPs/bkg
PYTHON=/home/export/sdurgut/micromamba/envs/foresee310/bin/python   # env's python directly (no shell hook needed)
cd "$WORKDIR"

: "${MODE:?set MODE=hard or MODE=soft}"
: "${NEV:?set NEV=<events per task>}"
: "${OUTDIR:?set OUTDIR=<output dir>}"
mkdir -p "$OUTDIR/json" "$OUTDIR/csv" "$OUTDIR/vbfcsv" logs

TASK=${SLURM_ARRAY_TASK_ID}
# distinct seed per (mode,task): keep hard and soft seed ranges from colliding
if [ "$MODE" = "hard" ]; then
    SEED=$((100000 + TASK))
else
    SEED=$((900000 + TASK))
fi
JSON="$OUTDIR/json/${MODE}_${TASK}.json"

DUMP_ARG=()
if [ "${DUMP:-0}" = "1" ]; then
    if [ "$MODE" = "soft" ]; then
        DUMP_ARG=(--dump-kl "$OUTDIR/csv/kl_${TASK}.csv")             # inclusive K_L sample
    elif [ "$MODE" = "hard" ]; then
        DUMP_ARG=(--dump-vbf-kl "$OUTDIR/vbfcsv/klvbf_${TASK}.csv")   # VBF-event background sample
    fi
fi

echo "task=$TASK mode=$MODE seed=$SEED nev=$NEV host=$(hostname)"
"$PYTHON" eps_vbf_pythia.py \
    --mode "$MODE" --n-events "$NEV" --seed "$SEED" \
    --pthatmin 80 --out-json "$JSON" --progress "${PROGRESS:-10000}" "${DUMP_ARG[@]}"
echo "done -> $JSON"
