#!/usr/bin/env bash
set -euo pipefail

# Convert all MadGraph LHE files under CMSRun3* process directories to CSV
# using tracking-ALPs/lhe_to_csv.py. Each output file is named with the process
# folder basename and the run folder (e.g. ..._run_01.csv).
#
# Layout expected:
#   ${ROOT}/CMSRun3_*/Events/run_*/unweighted_events.lhe[.gz]
#
# Root directory (first non-empty wins):
#   CMSRUN3_ROOT, or MG5_DIR, or default <repo>/tools/MG5_aMC_v3_5_13
#
# Other env vars:
#   OUT_DIR – output folder (default: <tracking-ALPs>/data/cmsrun3-csvs)
#   PYTHON  – interpreter (default: python3)
#
# Example – tree under tracking-ALPs/data/cmsrun3-madgraph:
#   CMSRUN3_ROOT=/path/to/tracking-ALPs/data/cmsrun3-madgraph \\
#     OUT_DIR=/path/to/tracking-ALPs/data/cmsrun3-madgraph-csvs \\
#     ./batch_lhe_to_csv_cmsrun3.sh
#
# Optional: MAX_EVENTS=10000 limits each file (passed as --max-events).

TRACKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TRACKING_DIR}/.." && pwd)"
LHE_TO_CSV="${TRACKING_DIR}/lhe_to_csv.py"

MG5_DIR="/Users/sena/grad_school/Research/ALP/tracking-ALPs/data/cmsrun3-madgraph"
OUT_DIR="${OUT_DIR:-${TRACKING_DIR}/data/cmsrun3-csvs}"
PYTHON="${PYTHON:-python3}"

extra_args=()
if [[ -n "${MAX_EVENTS:-}" ]]; then
  extra_args=(--max-events "${MAX_EVENTS}")
fi

if [[ ! -f "${LHE_TO_CSV}" ]]; then
  echo "ERROR: lhe_to_csv.py not found: ${LHE_TO_CSV}" >&2
  exit 2
fi

if [[ ! -d "${MG5_DIR}" ]]; then
  echo "ERROR: MG5 directory not found: ${MG5_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "ROOT (CMSRun3_* parent): ${MG5_DIR}"
echo "Output directory:        ${OUT_DIR}"
echo

n_ok=0
n_skip=0

# Use find + sort (not globs) so each directory is listed once and order is stable.
while IFS= read -r proc_dir; do
  proc_dir="${proc_dir%/}"
  [[ -d "${proc_dir}/Events" ]] || continue
  proc_base="$(basename "${proc_dir}")"

  while IFS= read -r run_dir; do
    run_dir="${run_dir%/}"
    [[ -d "${run_dir}" ]] || continue
    run_base="$(basename "${run_dir}")"

    lhe=""
    if [[ -f "${run_dir}/unweighted_events.lhe.gz" ]]; then
      lhe="${run_dir}/unweighted_events.lhe.gz"
    elif [[ -f "${run_dir}/unweighted_events.lhe" ]]; then
      lhe="${run_dir}/unweighted_events.lhe"
    else
      echo "SKIP: no unweighted_events.lhe(.gz) in ${run_dir}" >&2
      n_skip=$((n_skip + 1))
      continue
    fi

    out_csv="${OUT_DIR}/${proc_base}_${run_base}.csv"
    echo "==> ${lhe}"
    echo "    -> ${out_csv}"
    # Bash 3.2 + set -u: "${extra_args[@]}" errors when the array is empty.
    if [[ ${#extra_args[@]} -eq 0 ]]; then
      "${PYTHON}" "${LHE_TO_CSV}" "${lhe}" "${out_csv}"
    else
      "${PYTHON}" "${LHE_TO_CSV}" "${lhe}" "${out_csv}" "${extra_args[@]}"
    fi
    n_ok=$((n_ok + 1))
  done < <(find "${proc_dir}/Events" -maxdepth 1 -mindepth 1 -type d -name 'run_*' 2>/dev/null | LC_ALL=C sort)

done < <(find "${MG5_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'CMSRun3*' 2>/dev/null | LC_ALL=C sort)

echo "Done: ${n_ok} CSV file(s) written to ${OUT_DIR} (${n_skip} run dir(s) skipped)."
