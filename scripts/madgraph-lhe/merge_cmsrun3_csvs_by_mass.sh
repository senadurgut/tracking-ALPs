#!/usr/bin/env bash
set -euo pipefail

# Merge per-run CSV files into one CSV per mass point.
#
# Input filename pattern expected (examples):
#   CMSRun3_vbf_ax_ma_5p0000GeV_run_01.csv
#   CMSRun3_vbf_ax_ma_5p0000GeV_2_run_10.csv
#
# Mass key is taken from the substring between "_ma_" and "GeV"
# (e.g. "5p0000", "0p0100", ...). Files with the same mass key are merged.
#
# Merging is a plain concatenation (append files one after another).
#
# Env overrides:
#   IN_DIR  : directory containing per-run CSVs
#   OUT_DIR : directory to write merged CSVs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN_DIR="${IN_DIR:-${SCRIPT_DIR}/data/cmsrun3-csvs}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/data/cmsrun3-csvs-merged}"

if [[ ! -d "${IN_DIR}" ]]; then
  echo "ERROR: input directory not found: ${IN_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

tmp_list="$(mktemp)"
trap 'rm -f "${tmp_list}"' EXIT

# Build a sortable list: mass<TAB>runNumber<TAB>filepath
# (Bash 3.2 compatible; avoids associative arrays.)
found_any=0
for f in "${IN_DIR}"/*.csv; do
  [[ -f "${f}" ]] || continue
  base="$(basename "${f}")"

  # Skip already-merged outputs if you rerun.
  if [[ "${base}" == merged_ma_* ]]; then
    continue
  fi

  # Extract mass key: ..._ma_<MASS>GeV...
  # Note: sed backrefs are \1 (not \\1) on macOS.
  mass="$(printf '%s' "${base}" | sed -nE 's/.*_ma_([0-9]+p[0-9]+)GeV.*/\1/p')"
  if [[ -z "${mass}" ]]; then
    # Not a CMSRun3 mass-point CSV we recognize.
    continue
  fi

  # Extract run number: ..._run_XX.csv  (defaults to 0 if not found)
  run="$(printf '%s' "${base}" | sed -nE 's/.*_run_([0-9]+)\\.csv/\1/p')"
  if [[ -z "${run}" ]]; then
    run="0"
  fi

  printf '%s\t%08d\t%s\n' "${mass}" "${run}" "${f}" >> "${tmp_list}"
  found_any=1
done

if [[ "${found_any}" -eq 0 ]]; then
  echo "No matching CSV files found in: ${IN_DIR}" >&2
  echo "Expected names like: CMSRun3_*_ma_5p0000GeV*_run_01.csv" >&2
  exit 0
fi

LC_ALL=C sort "${tmp_list}" > "${tmp_list}.sorted"
mv -f "${tmp_list}.sorted" "${tmp_list}"

n_files="$(awk 'END {print NR+0}' "${tmp_list}")"
n_masses="$(cut -f1 "${tmp_list}" | LC_ALL=C sort -u | awk 'END {print NR+0}')"

echo "Input directory:  ${IN_DIR}"
echo "Output directory: ${OUT_DIR}"
echo "Found ${n_files} input file(s) across ${n_masses} mass point(s)."
echo

current_mass=""
out_file=""
n_out=0
n_in_total=0
mass_idx=0
idx_in_mass=0
files_this_mass=0

while IFS=$'\t' read -r mass run file; do
  [[ -n "${mass}" ]] || continue

  if [[ "${mass}" != "${current_mass}" ]]; then
    current_mass="${mass}"
    out_file="${OUT_DIR}/merged_ma_${mass}GeV.csv"
    : > "${out_file}"
    n_out=$((n_out + 1))
    mass_idx=$((mass_idx + 1))
    idx_in_mass=0
    files_this_mass="$(awk -F'\t' -v m="${mass}" '$1 == m { c++ } END { print c + 0 }' "${tmp_list}")"
    echo "==> [mass ${mass_idx}/${n_masses}] ma_${mass} GeV  (${files_this_mass} run file(s)) -> ${out_file}"
  fi

  idx_in_mass=$((idx_in_mass + 1))
  n_in_total=$((n_in_total + 1))
  echo "    [${n_in_total}/${n_files}] [run ${idx_in_mass}/${files_this_mass}] run_${run}  $(basename "${file}")"

  # Append without altering structure.
  cat "${file}" >> "${out_file}"
done < "${tmp_list}"

echo
echo "Done: merged ${n_in_total} input file(s) into ${n_out} output file(s)."
