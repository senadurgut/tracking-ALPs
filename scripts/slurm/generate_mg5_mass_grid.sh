#!/usr/bin/env bash
set -euo pipefail

# Generate MadGraph events for a grid of ALP masses, using an existing
# process directory as a template for all run settings.
#
# Template expected (already exists):
#   /Users/sena/grad_school/Research/ALP/tools/MG5_aMC_v3_5_13/trial_01_gev
#
# Output:
#   One new MG5 process directory per mass point under MG5_aMC_v3_5_13/
#
# Notes:
# - This script edits:
#     Cards/param_card.dat : BLOCK MASS entry 36 (ax) -> m_a
#     Cards/run_card.dat   : nevents -> 10000  (then uses multi_run to reach 1e6 total)
# - Running 1e6 unweighted events per mass point can take a long time.

MG5_DIR="/home/export/sdurgut/scratch/alps/tools/MG5_aMC_v3_5_13"
TEMPLATE_PROCESS="trial_01_gev"
TOTAL_NEVENTS="1000000"
NEVENTS_PER_RUN="10000"
N_RUNS="10"

MASS_POINTS=(
  0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
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
print(f"vbf_ax_ma_{s}GeV")
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

# Patch mass entry: "36 <value> # max"
param_txt = param.read_text()
lines = param_txt.splitlines(True)
out = []
patched = False
for ln in lines:
    m = re.match(r"^\s*36\s+([Ee0-9+\-\.]+)\s+#\s*max\b", ln)
    if m:
        # keep comment, replace value with scientific notation
        out.append(re.sub(r"^\s*36\s+[Ee0-9+\-\.]+", f"      36 {ma:.6e}", ln))
        patched = True
    else:
        out.append(ln)
if not patched:
    raise SystemExit("Failed to patch ALP mass: could not find '36 ... # max' in param_card.dat")
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

combine_lhe_runs() {
  local proc_dir="$1"
  local out_gz="$2"
  shift 2
  # remaining args: list of run directories (Events/run_XX)

  python3 - "$out_gz" "$@" <<'PY'
import gzip
import sys
from pathlib import Path

out_gz = Path(sys.argv[1])
run_dirs = [Path(p) for p in sys.argv[2:]]

in_files = []
for rd in run_dirs:
    for name in ("unweighted_events.lhe.gz", "unweighted_events.lhe"):
        p = rd / name
        if p.is_file():
            in_files.append(p)
            break
    else:
        raise SystemExit(f"Missing unweighted_events in {rd}")

def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")

header = []
events = []
footer = None

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

            # outside events
            if i == 0:
                header.append(line)
            else:
                pass

# Build header/footers properly: keep everything from first file up to (but excluding) </LesHouchesEvents>
full_header = "".join(header)
end_tag = "</LesHouchesEvents>"
if end_tag in full_header:
    pre, _post = full_header.split(end_tag, 1)
    header_text = pre
else:
    header_text = full_header

out_gz.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out_gz, "wt") as out:
    out.write(header_text)
    if not header_text.endswith("\n"):
        out.write("\n")
    for ev in events:
        out.write(ev)
        if not ev.endswith("\n"):
            out.write("\n")
    out.write(end_tag + "\n")
PY
}

generate_for_process() {
  local proc_dir="$1"
  local run_name="$2"

  # Robust large-sample generation:
  # run generate_events multiple times (100k each) with different seeds,
  # then merge the resulting LHEs into one combined file.
  (
    cd "$proc_dir"
    run_dirs=()
    for ((k=1; k<=N_RUNS; k++)); do
      seed=$(( (RANDOM << 16) + RANDOM + k ))
      echo "      run $k/$N_RUNS (iseed=$seed, nevents=$NEVENTS_PER_RUN)"
      set_seed "$proc_dir" "$seed"
      ./bin/generate_events -f
      # MG5 will create Events/run_XX sequentially; pick the newest one.
      last_run_dir="$(ls -1dt Events/run_* 2>/dev/null | head -n 1)"
      if [[ -z "$last_run_dir" ]]; then
        echo "ERROR: no Events/run_* directory created" >&2
        exit 1
      fi
      run_dirs+=("$last_run_dir")
    done

    out_dir="Events/${run_name}"
    mkdir -p "$out_dir"
    echo "      combining LHEs -> ${out_dir}/unweighted_events.lhe.gz"
    combine_lhe_runs "$proc_dir" "${out_dir}/unweighted_events.lhe.gz" "${run_dirs[@]}"
  )
}

echo "MG5 dir:      ${MG5_DIR}"
echo "Template:     ${TEMPLATE_PROCESS}"
echo "Events/mass:  ${TOTAL_NEVENTS}  (as ${N_RUNS} × ${NEVENTS_PER_RUN})"
echo "Mass points:  ${#MASS_POINTS[@]}"
echo

for ma in "${MASS_POINTS[@]}"; do
  proc_name="$(proc_name_for_mass "$ma")"
  proc_dir="${MG5_DIR}/${proc_name}"
  run_name="grid_${proc_name}"

  echo "==> m_a = ${ma} GeV  ->  ${proc_name}"

  if [[ -d "$proc_dir" ]]; then
    echo "    exists: ${proc_dir} (will reuse and re-patch cards)"
  else
    echo "    creating: ${proc_dir}"
    # Copy template process directory as starting point.
    # We remove prior event output if any.
    cp -R "${MG5_DIR}/${TEMPLATE_PROCESS}" "$proc_dir"
    rm -rf "${proc_dir}/Events" "${proc_dir}/HTML" "${proc_dir}/crossx.html" 2>/dev/null || true
  fi

  echo "    patching cards (m_a, nevents)..."
  patch_cards "$proc_dir" "$ma"

  echo "    generating events..."
  generate_for_process "$proc_dir" "$run_name"

  echo "    done: ${proc_name}"
  echo
done

echo "All mass points completed."

