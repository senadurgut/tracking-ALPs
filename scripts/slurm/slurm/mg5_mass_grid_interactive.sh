#!/usr/bin/env bash
# Interactive version of mg5_mass_grid_array.sh (no SLURM).
#
# Default: runs all 32 mass points one after another (same grid as SLURM array 0–31).
# Optional: pass a single index 0..31 to run only that point.
#
# Usage:
#   ./mg5_mass_grid_interactive.sh              # batch: all 32 masses, sequentially
#   ./mg5_mass_grid_interactive.sh --all        # same as no arguments
#   ./mg5_mass_grid_interactive.sh 5            # only grid index 5
#   MASS_INDEX=5 ./mg5_mass_grid_interactive.sh # only index 5 (no argv)
#
# Optional env: NEVENTS_PER_RUN, N_RUNS, MASS_POINTS_FILE, INTERACTIVE_JOB_ID

set -euo pipefail

MG5_DIR="/home/export/sdurgut/scratch/alps/tools/MG5_aMC_v3_5_13"
TEMPLATE_PROCESS="trial_01_gev"

# Total events per mass point = N_RUNS * NEVENTS_PER_RUN (kept <=1e6 per MG5 warning)
NEVENTS_PER_RUN="${NEVENTS_PER_RUN:-10000}"
N_RUNS="${N_RUNS:-1}"

# Replaces SLURM_JOB_ID for iseed mixing (override if you need reproducibility across shells)
INTERACTIVE_JOB_ID="${INTERACTIVE_JOB_ID:-$$}"

MASS_POINTS=(
  0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
  0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16
  0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24
  0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32
)

if [[ -n "${MASS_POINTS_FILE:-}" ]]; then
  mapfile -t MASS_POINTS < <(grep -v '^\s*$' "$MASS_POINTS_FILE" | grep -v '^\s*#')
fi

if [[ "${#MASS_POINTS[@]}" -ne 32 ]]; then
  echo "ERROR: need exactly 32 mass points, got ${#MASS_POINTS[@]}" >&2
  exit 2
fi

if [[ ! -d "${MG5_DIR}/${TEMPLATE_PROCESS}" ]]; then
  echo "ERROR: template process not found: ${MG5_DIR}/${TEMPLATE_PROCESS}" >&2
  exit 2
fi
if [[ ! -x "${MG5_DIR}/${TEMPLATE_PROCESS}/bin/generate_events" ]]; then
  echo "ERROR: template has no executable bin/generate_events" >&2
  exit 2
fi

proc_name_for_mass() {
  python3 - "$1" <<'PY'
import sys
m = float(sys.argv[1])
s = f"{m:.4f}".replace(".", "p")
print(f"CMSRun3_vbf_ax_ma_{s}GeV")
PY
}

patch_cards() {
  local proc_dir="$1"
  local ma="$2"
  local nevents="$3"
  python3 - "$proc_dir" "$ma" "$nevents" <<'PY'
import re, sys
from pathlib import Path

proc_dir = Path(sys.argv[1])
ma = float(sys.argv[2])
nevents = int(sys.argv[3])

param = proc_dir / "Cards" / "param_card.dat"
run = proc_dir / "Cards" / "run_card.dat"

ptxt = param.read_text().splitlines(True)
out, patched = [], False
for ln in ptxt:
    m = re.match(r"^\s*36\s+([Ee0-9+\-\.]+)\s+#\s*max\b", ln)
    if m:
        out.append(re.sub(r"^\s*36\s+[Ee0-9+\-\.]+", f"      36 {ma:.6e}", ln))
        patched = True
    else:
        out.append(ln)
if not patched:
    raise SystemExit("Failed to patch mass: no '36 ... # max' in param_card.dat")
param.write_text("".join(out))

rtxt = run.read_text().splitlines(True)
out, patched = [], False
for ln in rtxt:
    if re.search(r"=\s*nevents\b", ln):
        out.append(re.sub(r"^\s*\d+\s*=\s*nevents\b", f"  {nevents} = nevents", ln))
        patched = True
    else:
        out.append(ln)
if not patched:
    raise SystemExit("Failed to patch nevents in run_card.dat")
run.write_text("".join(out))
PY
}

set_seed() {
  local proc_dir="$1"
  local seed="$2"
  python3 - "$proc_dir" "$seed" <<'PY'
import re, sys
from pathlib import Path

proc_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
run = proc_dir / "Cards" / "run_card.dat"

rtxt = run.read_text().splitlines(True)
out, patched = [], False
for ln in rtxt:
    if re.search(r"=\s*iseed\b", ln):
        out.append(re.sub(r"^\s*\d+\s*=\s*iseed\b", f"  {seed} = iseed", ln))
        patched = True
    else:
        out.append(ln)
if not patched:
    raise SystemExit("Failed to patch iseed in run_card.dat")
run.write_text("".join(out))
PY
}

combine_lhe_runs() {
  local out_gz="$1"; shift
  python3 - "$out_gz" "$@" <<'PY'
import gzip, sys
from pathlib import Path

out_gz = Path(sys.argv[1])
run_dirs = [Path(p) for p in sys.argv[2:]]

def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")

in_files = []
for rd in run_dirs:
    for name in ("unweighted_events.lhe.gz", "unweighted_events.lhe"):
        p = rd / name
        if p.is_file():
            in_files.append(p)
            break
    else:
        raise SystemExit(f"Missing unweighted_events in {rd}")

header, events = [], []
for i, fpath in enumerate(in_files):
    with open_text(fpath) as fh:
        in_event = False
        buf = []
        for line in fh:
            if line.strip() == "<event>":
                in_event = True
                buf = [line]
                continue
            if in_event:
                buf.append(line)
                if line.strip() == "</event>":
                    events.append("".join(buf))
                    in_event = False
                continue
            if i == 0:
                header.append(line)

full_header = "".join(header)
end_tag = "</LesHouchesEvents>"
header_text = full_header.split(end_tag, 1)[0] if end_tag in full_header else full_header

out_gz.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out_gz, "wt") as out:
    out.write(header_text.rstrip("\n") + "\n")
    for ev in events:
        out.write(ev.rstrip("\n") + "\n")
    out.write(end_tag + "\n")
PY
}

run_one_mass_index() {
  local idx="$1"
  local ma="${MASS_POINTS[$idx]}"

  local proc_name
  proc_name="$(proc_name_for_mass "$ma")"
  local proc_dir="${MG5_DIR}/${proc_name}"
  local run_name="grid_${proc_name}"

  echo "[$(date)] interactive index=${idx} mass=${ma} proc=${proc_name}"
  echo "MG5_DIR=${MG5_DIR}"
  echo "template=${TEMPLATE_PROCESS}"
  echo "events per mass: $((N_RUNS * NEVENTS_PER_RUN)) (as ${N_RUNS} x ${NEVENTS_PER_RUN})"

  if [[ -d "$proc_dir" ]]; then
    echo "Reusing existing $proc_dir (will re-patch cards)"
  else
    echo "Creating $proc_dir from template"
    cp -R "${MG5_DIR}/${TEMPLATE_PROCESS}" "$proc_dir"
    rm -rf "${proc_dir}/Events" "${proc_dir}/HTML" "${proc_dir}/crossx.html" 2>/dev/null || true
  fi

  # Ensure the per-mass process has an up-to-date me5_configuration (esp. mg5_path),
  # even when reusing an existing directory created before you edited the template.
  if [[ -f "${MG5_DIR}/${TEMPLATE_PROCESS}/Cards/me5_configuration.txt" ]]; then
    cp -f "${MG5_DIR}/${TEMPLATE_PROCESS}/Cards/me5_configuration.txt" "${proc_dir}/Cards/me5_configuration.txt"
  fi

  patch_cards "$proc_dir" "$ma" "$NEVENTS_PER_RUN"

  cd "$proc_dir"
  mkdir -p Events

  local run_dirs=()
  local k
  for ((k=1; k<=N_RUNS; k++)); do
    local seed=$(( (INTERACTIVE_JOB_ID % 1000000000) + (idx * 1000) + k ))
    echo "  run $k/$N_RUNS (iseed=$seed)"
    set_seed "$proc_dir" "$seed"

    ./bin/generate_events -f

    local last_run_dir
    last_run_dir="$(ls -1dt Events/run_* 2>/dev/null | head -n 1 || true)"
    if [[ -z "$last_run_dir" ]]; then
      echo "ERROR: no Events/run_* directory created" >&2
      exit 1
    fi
    run_dirs+=("$last_run_dir")
  done

  local out_dir="Events/${run_name}"
  mkdir -p "$out_dir"
  echo "Combining -> ${out_dir}/unweighted_events.lhe.gz"
  combine_lhe_runs "${out_dir}/unweighted_events.lhe.gz" "${run_dirs[@]}"

  local lhe_out="${proc_dir}/${out_dir}/unweighted_events.lhe.gz"
  echo ""
  echo "================================================================"
  echo "  Mass point DONE  |  index ${idx}/31  |  ma = ${ma} GeV"
  echo "  Output: ${lhe_out}"
  echo "  $(date -Iseconds)"
  echo "================================================================"
  echo ""
}

run_all_masses() {
  echo "Interactive batch: running all 32 mass points sequentially (no SLURM)."
  echo "Started: $(date -Iseconds)"
  echo ""
  local i
  for i in $(seq 0 31); do
    echo "----------------------------------------------------------------"
    echo "  BATCH: starting mass index ${i}/31  |  $(date -Iseconds)"
    echo "----------------------------------------------------------------"
    run_one_mass_index "$i"
  done
  echo ""
  echo "================================================================"
  echo "  All 32 mass points finished."
  echo "  Completed: $(date -Iseconds)"
  echo "================================================================"
}

usage() {
  echo "Usage: $0                 # batch: all 32 masses, sequentially" >&2
  echo "       $0 --all           # same as above" >&2
  echo "       $0 <index>         # single mass, index 0..31" >&2
  echo "Env: MASS_INDEX (0..31) with no argv for a single point; NEVENTS_PER_RUN, N_RUNS, MASS_POINTS_FILE, INTERACTIVE_JOB_ID" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Default: full batch of 32. Single index if argv is a number, or MASS_INDEX with no argv.
if [[ -z "${1:-}" ]]; then
  if [[ -n "${MASS_INDEX:-}" ]]; then
    if ! [[ "${MASS_INDEX}" =~ ^[0-9]+$ ]] || [[ "${MASS_INDEX}" -lt 0 || "${MASS_INDEX}" -gt 31 ]]; then
      echo "ERROR: MASS_INDEX must be integer 0..31, got: ${MASS_INDEX}" >&2
      exit 2
    fi
    run_one_mass_index "${MASS_INDEX}"
  else
    run_all_masses
  fi
  exit 0
fi

if [[ "${1}" == "--all" ]]; then
  run_all_masses
  exit 0
fi

if ! [[ "${1}" =~ ^[0-9]+$ ]] || [[ "${1}" -lt 0 || "${1}" -gt 31 ]]; then
  echo "ERROR: expected index 0..31, --all, or no arguments for full batch; got: ${1}" >&2
  usage
  exit 2
fi

run_one_mass_index "${1}"
