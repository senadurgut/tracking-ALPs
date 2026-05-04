#!/usr/bin/env bash
#SBATCH --job-name=mg5_alp_grid
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --qos=cpu
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

MG5_DIR="/home/export/sdurgut/scratch/alps/tools/MG5_aMC_v3_5_13"
TEMPLATE_PROCESS="trial_01_gev"

# Total events per mass point = N_RUNS * NEVENTS_PER_RUN (kept <=1e6 per MG5 warning)
# Default "smoke test": 10k events per mass point (1 run).
NEVENTS_PER_RUN="${NEVENTS_PER_RUN:-10000}"
N_RUNS="${N_RUNS:-1}"

# Option A (default): hardcode the missing masses here
MASS_POINTS=(
  3.0 4.0 6.0 7.0 8.0 9.0
)

# Option B: provide a file with one mass per line, and export MASS_POINTS_FILE=/path/to/masses.txt
if [[ -n "${MASS_POINTS_FILE:-}" ]]; then
  mapfile -t MASS_POINTS < <(grep -v '^\s*$' "$MASS_POINTS_FILE" | grep -v '^\s*#')
fi

if [[ "${#MASS_POINTS[@]}" -lt 1 ]]; then
  echo "ERROR: MASS_POINTS is empty" >&2
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

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID not set (did you run outside Slurm?)" >&2
  exit 2
fi
if [[ "$SLURM_ARRAY_TASK_ID" -lt 0 || "$SLURM_ARRAY_TASK_ID" -ge "${#MASS_POINTS[@]}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${#MASS_POINTS[@]} mass points" >&2
  exit 2
fi

ma="${MASS_POINTS[$SLURM_ARRAY_TASK_ID]}"

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

proc_name="$(proc_name_for_mass "$ma")"
proc_dir="${MG5_DIR}/${proc_name}"
run_name="grid_${proc_name}"

echo "[$(date)] SLURM job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} mass=${ma} proc=${proc_name}"
echo "MG5_DIR=${MG5_DIR}"
echo "template=${TEMPLATE_PROCESS}"
echo "events per mass: $((N_RUNS * NEVENTS_PER_RUN)) (as ${N_RUNS} x ${NEVENTS_PER_RUN})"

# Create a clean per-mass process dir (A2: always delete + re-copy).
# This avoids mixed compiler artifacts (e.g. gfortran .mod version mismatches) on clusters.
if [[ -d "$proc_dir" ]]; then
  echo "Removing existing $proc_dir for a clean rebuild"
  rm -rf "$proc_dir"
fi
echo "Creating $proc_dir from template"
cp -R "${MG5_DIR}/${TEMPLATE_PROCESS}" "$proc_dir"

# If the template was copied from another machine, it may contain precompiled
# Fortran artifacts (*.mod/*.o/*.a) built with a different gfortran. Those cause:
#   "Cannot read module file ... created by a different version of GNU Fortran".
# Scrub them so each Slurm task recompiles cleanly on the compute node.
shopt -s globstar nullglob
rm -f \
  "$proc_dir"/**/*.mod \
  "$proc_dir"/**/*.o \
  "$proc_dir"/**/*.a \
  "$proc_dir"/**/*.so \
  "$proc_dir"/**/*.dylib \
  "$proc_dir"/**/*.exe \
  "$proc_dir"/**/*.pyc
shopt -u globstar nullglob

# Cluster builds typically have libstdc++ but not libc++.
# Ensure the copied template doesn't try to link against -lc++.
if [[ -f "$proc_dir/Source/make_opts" ]]; then
  python3 - "$proc_dir/Source/make_opts" <<'PY'
from pathlib import Path

p = Path(__file__).parent  # unused, keep for robustness
path = Path(__import__("sys").argv[1])
txt = path.read_text()
txt2 = txt
txt2 = txt2.replace("DEFAULT_CPP_COMPILER=clang", "DEFAULT_CPP_COMPILER=g++")
txt2 = txt2.replace("STDLIB=-lc++", "STDLIB=-lstdc++")
# clang-only flag: safe to blank on Linux/g++
txt2 = txt2.replace("STDLIB_FLAG=-stdlib=libc++", "STDLIB_FLAG=")
if txt2 != txt:
    path.write_text(txt2)
PY
fi

patch_cards "$proc_dir" "$ma" "$NEVENTS_PER_RUN"

cd "$proc_dir"
mkdir -p Events

run_dirs=()
for ((k=1; k<=N_RUNS; k++)); do
  # deterministic-ish seed: job id + array id + iteration
  seed=$(( (SLURM_JOB_ID % 1000000000) + (SLURM_ARRAY_TASK_ID * 1000) + k ))
  echo "  run $k/$N_RUNS (iseed=$seed)"
  set_seed "$proc_dir" "$seed"

  ./bin/generate_events -f

  last_run_dir="$(ls -1dt Events/run_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$last_run_dir" ]]; then
    echo "ERROR: no Events/run_* directory created" >&2
    exit 1
  fi
  run_dirs+=("$last_run_dir")
done

out_dir="Events/${run_name}"
mkdir -p "$out_dir"
echo "Combining -> ${out_dir}/unweighted_events.lhe.gz"
combine_lhe_runs "${out_dir}/unweighted_events.lhe.gz" "${run_dirs[@]}"

echo "DONE mass=${ma} output=${proc_dir}/${out_dir}/unweighted_events.lhe.gz"