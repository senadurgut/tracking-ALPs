#!/usr/bin/env bash
set -euo pipefail

# Generate MadGraph events for a grid of ALP masses, using an existing
# process directory as a template for all run settings.
#
# Template expected (already exists):
#   /Users/sena/grad_school/Research/ALP/tools/MG5_aMC_v3_5_13/CMS
#
# Output:
#   One new MG5 process directory per mass point under MG5_aMC_v3_5_13/
#
# Notes:
# - This script edits:
#     Cards/param_card.dat : BLOCK MASS entry 36 (ax) -> m_a
#     Cards/run_card.dat   : nevents
# - On rerun, if a directory already exists for a mass point, this script
#   creates a new suffixed directory like _2, _3, etc.
# - No merging is done; MG5's Events/run_* directories are kept as-is.

MG5_DIR="/Users/sena/grad_school/Research/ALP/tools/MG5_aMC_v3_5_13"
TEMPLATE_PROCESS="CMS"
TOTAL_NEVENTS="100000"
NEVENTS_PER_RUN="10000"
N_RUNS="10"
MG5_MAX_SEED=$(( 30081 * 30081 ))   # MG5 hard limit: iseed <= 30081^2

MASS_POINTS=(
  0.04 0.05 0.06 0.07 0.08 0.09 0.1
  0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 5.0 10.0
)

if [[ ! -d "${MG5_DIR}/${TEMPLATE_PROCESS}" ]]; then
  echo "ERROR: template process not found: ${MG5_DIR}/${TEMPLATE_PROCESS}" >&2
  exit 2
fi

if [[ ! -x "${MG5_DIR}/${TEMPLATE_PROCESS}/bin/generate_events" ]]; then
  echo "ERROR: template has no executable bin/generate_events" >&2
  echo "Path checked: ${MG5_DIR}/${TEMPLATE_PROCESS}/bin/generate_events" >&2
  exit 2
fi

proc_name_for_mass() {
  # Convert e.g. 0.1 -> 0p1000, 10.0 -> 10p0000
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

  python3 - "$proc_dir" "$ma" "$NEVENTS_PER_RUN" <<'PY'
import re
import sys
from pathlib import Path

proc_dir = Path(sys.argv[1])
ma = float(sys.argv[2])
nevents = int(sys.argv[3])

param = proc_dir / "Cards" / "param_card.dat"
run = proc_dir / "Cards" / "run_card.dat"

if not param.is_file():
    raise SystemExit(f"param_card.dat not found: {param}")
if not run.is_file():
    raise SystemExit(f"run_card.dat not found: {run}")

# Patch mass entry: "36 <value> # Max"
param_txt = param.read_text()
lines = param_txt.splitlines(True)
out = []
patched = False
for ln in lines:
    m = re.match(r"^\s*36\s+([Ee0-9+\-\.]+)\s+#\s*Max\b", ln, flags=re.IGNORECASE)
    if m:
        out.append(re.sub(r"^\s*36\s+[Ee0-9+\-\.]+", f"      36 {ma:.6e}", ln))
        patched = True
    else:
        out.append(ln)
if not patched:
    raise SystemExit("Failed to patch ALP mass: could not find '36 ... # Max' in param_card.dat")
param.write_text("".join(out))

# Patch nevents in run_card: "<int> = nevents ! ..."
run_txt = run.read_text()
run_lines = run_txt.splitlines(True)
out = []
patched = False
for ln in run_lines:
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
import re
import sys
from pathlib import Path

proc_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
run = proc_dir / "Cards" / "run_card.dat"
txt = run.read_text()
lines = txt.splitlines(True)
out = []
patched = False
for ln in lines:
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

generate_for_process() {
  local proc_dir="$1"

  (
    cd "$proc_dir"
    mkdir -p Events HTML
    for ((k=1; k<=N_RUNS; k++)); do
      seed=$(( (RANDOM * 32768 + RANDOM + k) % MG5_MAX_SEED + 1 ))
      echo "      run $k/$N_RUNS (iseed=$seed, nevents=$NEVENTS_PER_RUN)"
      set_seed "$proc_dir" "$seed"
      ./bin/generate_events -f
    done
  )
}

echo "MG5 dir:      ${MG5_DIR}"
echo "Template:     ${TEMPLATE_PROCESS}"
echo "Events/mass:  ${TOTAL_NEVENTS}  (as ${N_RUNS} × ${NEVENTS_PER_RUN})"
echo "Mass points:  ${#MASS_POINTS[@]}"
echo

for ma in "${MASS_POINTS[@]}"; do
  base_proc_name="$(proc_name_for_mass "$ma")"
  proc_name="$base_proc_name"
  suffix=2
  while [[ -d "${MG5_DIR}/${proc_name}" ]]; do
    proc_name="${base_proc_name}_${suffix}"
    ((suffix++))
  done
  proc_dir="${MG5_DIR}/${proc_name}"

  echo "==> m_a = ${ma} GeV  ->  ${proc_name}"
  echo "    creating: ${proc_dir}"

  cp -R "${MG5_DIR}/${TEMPLATE_PROCESS}" "$proc_dir"
  rm -rf "${proc_dir}/Events" "${proc_dir}/HTML" "${proc_dir}/crossx.html" 2>/dev/null || true

  echo "    patching cards (m_a, nevents)..."
  patch_cards "$proc_dir" "$ma"

  echo "    generating events..."
  generate_for_process "$proc_dir"

  echo "    done: ${proc_name}"
  echo
done

echo "All mass points completed."