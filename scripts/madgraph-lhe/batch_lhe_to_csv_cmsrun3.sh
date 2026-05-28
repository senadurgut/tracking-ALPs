#!/usr/bin/env bash
set -euo pipefail

# Convert all MadGraph LHE files under CMSRun3* process directories to CSV
# using tracking-ALPs/lhe_to_csv.py. Each output file is named with the process
# folder basename and the run folder (e.g. ..._run_01.csv).
#
# Layout expected (either is fine; both are picked up):
#   ${ROOT}/CMSRun3_*/Events/grid*/unweighted_events.lhe[.gz]
#   ${ROOT}/CMSRun3_*/Events/run_*/unweighted_events.lhe[.gz]
#
# Root directory (first non-empty wins):
#   CMSRUN3_ROOT, or MG5_DIR, or default <repo>/tools/MG5_aMC_v3_5_13
#
# Other env vars:
#   OUT_DIR – output folder (default: <tracking-ALPs>/data/cmsrun3-csvs)
#   PYTHON  – interpreter (default: python3)
#   MAX_EVENTS – if set *and* CUMULATIVE_MAX_EVENTS is unset: pass --max-events to
#                 *every* run (caps each LHE file independently).
#   CUMULATIVE_MAX_EVENTS – if set (e.g. 100000): **for each** CMSRun3_* mass-point
#                          directory separately, walk Events/{grid*,run_*} in order until the
#                          events written for **that** mass reach this budget (then
#                          continue to the next mass). Not a single cap across all masses.
#                          Each call uses --max-events REMAINING; last run file may be
#                          truncated. Ignores MAX_EVENTS.
#
# Example – tree under tracking-ALPs/data/cmsrun3-madgraph:
#   CMSRUN3_ROOT=/path/to/tracking-ALPs/data/cmsrun3-madgraph \\
#     OUT_DIR=/path/to/tracking-ALPs/data/cmsrun3-madgraph-csvs \\
#     ./batch_lhe_to_csv_cmsrun3.sh
#
# Example – 100k events per mass point (per CMSRun3_* folder), summed over its run_* dirs:
#   CUMULATIVE_MAX_EVENTS=100000 ./batch_lhe_to_csv_cmsrun3.sh
#
# Example – only convert selected mass point(s):
#   ./batch_lhe_to_csv_cmsrun3.sh 20p0000
#   ./batch_lhe_to_csv_cmsrun3.sh 20p0000 30p0000
CUMULATIVE_MAX_EVENTS=40000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKING_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${TRACKING_DIR}"
LHE_TO_CSV="${TRACKING_DIR}/lhe_to_csv.py"

MG5_DIR="${CMSRUN3_ROOT:-${MG5_DIR:-/home/export/sdurgut/scratch/alps/tools/MG5_aMC_v3_5_13}}"
OUT_DIR="${OUT_DIR:-${TRACKING_DIR}/data/cmsrun3-csvs}"
PYTHON="${PYTHON:-python3}"
MASS_FILTERS=("$@")

extra_args=()
if [[ -z "${CUMULATIVE_MAX_EVENTS:-}" && -n "${MAX_EVENTS:-}" ]]; then
  extra_args=(--max-events "${MAX_EVENTS}")
fi

cumulative_mode=0
if [[ -n "${CUMULATIVE_MAX_EVENTS:-}" ]]; then
  cumulative_mode=1
  if ! [[ "${CUMULATIVE_MAX_EVENTS}" =~ ^[0-9]+$ ]] || [[ "${CUMULATIVE_MAX_EVENTS}" -le 0 ]]; then
    echo "ERROR: CUMULATIVE_MAX_EVENTS must be a positive integer, got: ${CUMULATIVE_MAX_EVENTS}" >&2
    exit 2
  fi
  echo "Per-mass event budget: ${CUMULATIVE_MAX_EVENTS} (each CMSRun3_* folder fills up to this, then next mass)"
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
if [[ "${#MASS_FILTERS[@]}" -gt 0 ]]; then
  echo "Mass filters:            ${MASS_FILTERS[*]}"
fi
echo

n_ok=0
n_skip=0

matches_mass_filter() {
  local proc_base="$1"
  local mass_arg token expected

  if [[ "${#MASS_FILTERS[@]}" -eq 0 ]]; then
    return 0
  fi

  for mass_arg in "${MASS_FILTERS[@]}"; do
    token="${mass_arg}"
    token="${token#CMSRun3_vbf_ax_ma_}"
    token="${token#grid_CMSRun3_vbf_ax_ma_}"
    token="${token%.csv}"
    token="${token%GeV}"
    expected="CMSRun3_vbf_ax_ma_${token}GeV"

    if [[ "${proc_base}" == "${expected}" ]]; then
      return 0
    fi
  done

  return 1
}

# Use find + sort (not globs) so each directory is listed once and order is stable.
while IFS= read -r proc_dir; do
  proc_dir="${proc_dir%/}"
  [[ -d "${proc_dir}/Events" ]] || continue
  proc_base="$(basename "${proc_dir}")"
  if ! matches_mass_filter "${proc_base}"; then
    continue
  fi

  cumulative_total=0
  if [[ "${cumulative_mode}" -eq 1 ]]; then
    echo ""
    echo "--- ${proc_base}: up to ${CUMULATIVE_MAX_EVENTS} events (this mass point only) ---"
  fi

  while IFS= read -r run_dir; do
    run_dir="${run_dir%/}"
    [[ -d "${run_dir}" ]] || continue
    run_base="$(basename "${run_dir}")"

    if [[ "${cumulative_mode}" -eq 1 ]]; then
      if [[ "${cumulative_total}" -ge "${CUMULATIVE_MAX_EVENTS}" ]]; then
        echo "Budget reached for ${proc_base} (${cumulative_total} >= ${CUMULATIVE_MAX_EVENTS}); next mass point."
        break
      fi
    fi

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
    if [[ "${cumulative_mode}" -eq 1 ]]; then
      remaining=$((CUMULATIVE_MAX_EVENTS - cumulative_total))
      if [[ "${remaining}" -le 0 ]]; then
        echo "Budget reached for ${proc_base}; next mass point."
        break
      fi
      _conv_out=$("${PYTHON}" "${LHE_TO_CSV}" "${lhe}" "${out_csv}" --max-events "${remaining}" 2>&1)
      _conv_ec=$?
      printf '%s\n' "${_conv_out}"
      if [[ "${_conv_ec}" -ne 0 ]]; then
        exit "${_conv_ec}"
      fi
      # lhe_to_csv.py prints: Written N events to ...
      _n_written=$(printf '%s\n' "${_conv_out}" | sed -n 's/^Written \([0-9][0-9]*\) events.*/\1/p' | tail -n 1)
      if [[ -z "${_n_written}" ]]; then
        echo "ERROR: could not parse event count from lhe_to_csv output." >&2
        exit 3
      fi
      cumulative_total=$((cumulative_total + _n_written))
      echo "    cumulative events so far: ${cumulative_total} / ${CUMULATIVE_MAX_EVENTS}"
    elif [[ ${#extra_args[@]} -eq 0 ]]; then
      "${PYTHON}" "${LHE_TO_CSV}" "${lhe}" "${out_csv}"
    else
      "${PYTHON}" "${LHE_TO_CSV}" "${lhe}" "${out_csv}" "${extra_args[@]}"
    fi
    n_ok=$((n_ok + 1))
  done < <(find "${proc_dir}/Events" -maxdepth 1 -mindepth 1 -type d \( -name 'grid*' -o -name 'run_*' \) 2>/dev/null | LC_ALL=C sort)

done < <(find "${MG5_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'CMSRun3*' 2>/dev/null | LC_ALL=C sort)

echo "Done: ${n_ok} CSV file(s) written to ${OUT_DIR} (${n_skip} run dir(s) skipped)."
if [[ "${cumulative_mode}" -eq 1 ]]; then
  echo "Per-mass budget was ${CUMULATIVE_MAX_EVENTS} events per CMSRun3_* directory (each mass independently)."
fi
