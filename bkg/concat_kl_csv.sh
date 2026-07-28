#!/bin/bash
# Concatenate the per-task K_L CSV chunks (out/csv/kl_*.csv) into one file,
# keeping a single header row.
#
# Usage:  ./concat_kl_csv.sh [CSV_DIR] [OUT_FILE]
#   defaults: CSV_DIR=out/csv   OUT_FILE=kl_sample.csv
set -euo pipefail

CSV_DIR="${1:-data/out/csv}"
OUT="${2:-data/kl_sample.csv}"

shopt -s nullglob
chunks=("$CSV_DIR"/klvbf*.csv)
if [ "${#chunks[@]}" -eq 0 ]; then
    echo "no klvbf*.csv chunks found in $CSV_DIR" >&2
    exit 1
fi

# header from the first chunk, then all data rows (skip each chunk's header)
head -n 1 "${chunks[0]}" > "$OUT"
tail -q -n +2 "${chunks[@]}" >> "$OUT"

rows=$(( $(wc -l < "$OUT") - 1 ))
echo "merged ${#chunks[@]} chunks -> $OUT  (${rows} K_L rows)"
